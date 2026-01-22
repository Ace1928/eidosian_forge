import pandas as pd
from faker import Faker

# Load the dataset
data = pd.read_csv("dataset.csv")

# Create a Faker instance
faker = Faker()

# Define the columns to mask
columns_to_mask = ["name", "email", "phone"]

# Mask the specified columns
for column in columns_to_mask:
    data[column] = data[column].apply(
        lambda x: (
            faker.name()
            if column == "name"
            else faker.email() if column == "email" else faker.phone_number()
        )
    )

# Save the masked dataset
data.to_csv("masked_dataset.csv", index=False)
