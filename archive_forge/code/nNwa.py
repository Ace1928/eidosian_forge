import requests
from bs4 import BeautifulSoup

# URL of the e-commerce website
url = "https://www.example.com/products"

# Send a GET request to the URL
response = requests.get(url)

# Create a BeautifulSoup object
soup = BeautifulSoup(response.content, "html.parser")

# Find all the product elements
products = soup.find_all("div", class_="product")

# Extract information from each product
for product in products:
    name = product.find("h3", class_="product-name").text
    price = product.find("span", class_="product-price").text
    description = product.find("p", class_="product-description").text

    print(f"Product: {name}")
    print(f"Price: {price}")
    print(f"Description: {description}")
    print("---")
