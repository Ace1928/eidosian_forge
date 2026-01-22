import great_expectations as ge

# Load the dataset
data = ge.read_csv("/home/lloyd/Downloads/exampledata/example.csv")

# Define expectations
expectations = [
    {
        "expectation_type": "expect_column_values_to_not_be_null",
        "kwargs": {"column": "price"},
    },
    {
        "expectation_type": "expect_column_values_to_be_between",
        "kwargs": {"column": "price", "min_value": 0.01, "max_value": 100},
    },
    {
        "expectation_type": "expect_column_values_to_be_in_set",
        "kwargs": {"column": "colour", "value_set": ["red", "green", "blue"]},
    },
]

# Validate the dataset against expectations
validation_result = data.validate(expectations)

# Print validation results
print(validation_result)

# Generate a validation report
validation_result.(
    "/home/lloyd/Downloads/exampledata/validation_report.html"
)
