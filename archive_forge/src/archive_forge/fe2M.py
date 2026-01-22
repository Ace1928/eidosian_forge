import pandas as pd

# Load the dataset
data = pd.read_csv("dataset.csv")

# Define validation rules
rules = {
    "column1": {"min": 0, "max": 100},
    "column2": {"allowed_values": ["A", "B", "C"]},
    "column3": {"not_null": True},
}

# Perform data validation
for column, rule in rules.items():
    if "min" in rule:
        invalid_data = data[data[column] < rule["min"]]
        if not invalid_data.empty:
            print(f"Validation failed for {column}: Values below minimum threshold.")

    if "max" in rule:
        invalid_data = data[data[column] > rule["max"]]
        if not invalid_data.empty:
            print(f"Validation failed for {column}: Values above maximum threshold.")

    if "allowed_values" in rule:
        invalid_data = data[~data[column].isin(rule["allowed_values"])]
        if not invalid_data.empty:
            print(f"Validation failed for {column}: Invalid values found.")

    if "not_null" in rule and rule["not_null"]:
        null_data = data[data[column].isnull()]
        if not null_data.empty:
            print(f"Validation failed for {column}: Null values found.")

print("Data validation completed.")
