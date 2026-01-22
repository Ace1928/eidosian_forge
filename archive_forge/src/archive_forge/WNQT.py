import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the dataset
data = pd.read_csv("dataset.csv")

# Handle missing values
imputer = SimpleImputer(strategy="mean")
data_imputed = imputer.fit_transform(data)

# Encode categorical variables
encoder = OneHotEncoder(handle_unknown="ignore")
data_encoded = encoder.fit_transform(data_imputed[:, :-1])

# Scale numerical features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Create the preprocessed dataset
preprocessed_data = pd.DataFrame(data_scaled)
