# Beutiful Soup Scraping
import requests
from bs4 import BeautifulSoup

# Send a GET request to the website
url = "https://www.example.com/news"
response = requests.get(url)
# Create a BeautifulSoup object and parse the HTML
soup = BeautifulSoup(response.content, "html.parser")
# Find all the article titles
titles = soup.find_all("h2", class_="article-title")
# Print the titles
for title in titles:
    print(title.text.strip())


# Selenium
from selenium import webdriver
from selenium.webdriver.common.by import By

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()
# Navigate to the login page
driver.get("https://www.example.com/login")
# Find the username and password input fields and enter the credentials
username_field = driver.find_element(By.ID, "username")
username_field.send_keys("your_username")
password_field = driver.find_element(By.ID, "password")
password_field.send_keys("your_password")
# Find and click the login button
login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')
login_button.click()
# Close the browser
driver.quit()
