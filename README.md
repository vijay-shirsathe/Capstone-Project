### Radio based Fingerprinting ML Algorithm for indoor localization

**Author**
Vijay Shirsathe

#### Executive summary
Using wireless communication signals for high precision indoor localization/positioning is a problem of great interest and has a wide variety of applications. It has traditionally been tackled with geometrical algorithms. However, research has shown that the ML techniques can improve the accuracy of positioning algorithms. This project explores training ML Models using "RF fingerprinting" dataset that can precisely determine a location of an object / tag based on its unique "RF fingerprint".

#### Rationale
Traditional GPS fails to operate effectively within buildings, tunnels, or complex structures. Having high-accuracy, real-time location data (often at sub-meter or centimeter-level) enables, optimizes, and secures operations in environments where knowing "where" is as important as "what". Here are a few example applications - Locating items/inventory in warehouses, tracking patient flow in hospitals, location based advertising/coupon delivery in malls, assisting first responders in navigating dangerous indoor environment during emergency operations and so on.

#### Research Question
Can we create a machine learning model for a given indoor environment that can accurately ouput a position/location of an entity based on the "RF fingerprint" data collected by the radio tag/device attached to the entity?

#### Data Sources
Thanks to Fraunhofer Institute for Integrated Circuits IIS for making available a great data source below which is in HDF5 file format. This project utilized the UWB dataset.  
https://www.iis.fraunhofer.de/en/ff/lv/dataanalytics/pos/fingerprinting-dataset-for-positioning.html

Here is the description of the data reproduced from the link above: "We designed a complex indoor environment with walls, that reflect radio signals on the inner side (iron surface) and absorb them at the outside (black surface). The transceivers, indicated as green dots, are placed at the edges of the recording area. The walls, indicated in red, are placed to block the line-of-sight (LoS) between the anchors and the robot platform, which causes ranging errors of the UWB radio system leading to high localization errors using classical positioning approaches." 

Please see the picture below (again, reproduced from the link above). One burst of data sample includes the six synchronized channel impulse responses (CIRs) and time-of-flight (ToF) measurements along with the "ground truth" precise position within the two dimentional rectangular indoor space. Dataset contains 143920 unique data samples.

<img width="644" height="499" alt="image" src="https://github.com/user-attachments/assets/0f159f69-57bb-4ef4-a4f1-0bcb5240cdf6" />

#### Methodology
What methods are you using to answer the question?
We applied EDA techniques on the dataset. After studying the structure of the HDF5 data, we converted it into a Pandas dataframe. We then developed understanding of each feature / column as well as rows/data samples. We developed several insights as listed below:
* Original HDF5 dataconsisted of the following datasets (sizes listed next to each dataset):
A_ID: (143920, 1)
B_ID: (143920, 1)
CIR_i: (143920, 366)
CIR_r: (143920, 366)
POS_x: (143920, 1)
POS_y: (143920, 1)
TD: (143920, 1)
TD_OFFSET: (143920, 1)
TIME_STAMP: (143920, 1)
* A_ID represents one of the six transievers (green dots) in the diagram above. B_ID represents signal recording tag at a particular position (POS_X, POS_Y).
* 366 CIR_r + 366 CIR_i samples for each (A_ID, B_ID) pair represent the "RF Fingerprint" recorded by B_ID for the signal transmitted by A_ID
* Since we are only interested in the magnitude of the CIR vector, we decided to replace 366 CIR_r & 366 CIR_i samples by 366 CIR_t samples by using formula CIR_t = sqrt (CIR_r**2 + CIR_i**2)
* Although there are six transmitters (A_IDs), not all locations can receive signals from all six A_IDs because of the blockage (walls shown in the diagram above). Therefore, not all B_IDs show up in six rows (one row per A_ID)
* For the "missing" A_IDs, we decided to use zeros for the CIR_t samples.
* To prepare this data for training ML algorithm, we needed to "flatten" it such that there was a single row per B_ID that consisted of the complete RF fingerprint seen by that B_ID across ALL six A_IDs. This means the total number of CIR_t columns for each B_ID increased to 366*6 = 2196. We then applied standardscaler on TD & TD_OFFSET data and normalized CIR_t data.
* Thus the dataframe ended up with 2198 features (X): TD, TD_OFFSET 2196 CIR_t and 2 Labels (y): POS_X and POS_Y

At this stage, we decided to apply MLP algorithm on this data with 4 hidden layers and with mean squared positioning error as the loss function.

#### Results
MLP algorithm is computationally intensive. It took over 5 hours to run on a x86 HP laptop (no GPU). It yielded RMSE of sqrt (452152) = 672 cm = 6.72 m. This is still far from the desired positioning accuracy of <= 1 m. Thus, we need to explore more otions (different MLP configurations, other algorithms such as 1D-CNN, KNN or Decision Trees). We may also look

#### Next steps
To further improve positioning accuracy, we would explore other otions such as different MLP configurations as well as other ML algorithms such as 1D-CNN, KNN or Decision Trees. We would also look into applying ensemble techinques.

#### Outline of project

- https://github.com/vijay-shirsathe/CapstoneProject/blob/main/PositioningByFingerPrinting.ipynb

##### Contact and Further Information





