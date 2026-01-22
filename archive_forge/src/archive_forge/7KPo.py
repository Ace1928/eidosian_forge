import pandas as pd


# Read the CSV file into a DataFrame
df = pd.read_csv("sales_data.csv")
# Display the first few rows of the DataFrame
print(df.head())
# Calculate total sales by product category
sales_by_category = df.groupby("Category")["Sales"].sum()
print(sales_by_category)
# Filter the DataFrame to include only sales above a certain threshold
high_sales = df[df["Sales"] > 1000]
print(high_sales)
# Create a new column based on a condition
df["Discount"] = df["Price"].apply(lambda x: 0.1 if x > 50 else 0)
print(df.head())
# Save the modified DataFrame to a new CSV file
df.to_csv("updated_sales_data.csv", index=False)
