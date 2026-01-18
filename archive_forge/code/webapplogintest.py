from selenium import webdriver
from selenium.webdriver.common.by import By

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to the login page
driver.get("https://example.com/login")

# Find the username and password fields and enter the credentials
username_field = driver.find_element(By.NAME, "username")
username_field.send_keys("your_username")

password_field = driver.find_element(By.NAME, "password")
password_field.send_keys("your_password")

# Find and click the login button
login_button = driver.find_element(By.XPATH, '//button[@type="submit"]')
login_button.click()

# Verify successful login
welcome_message = driver.find_element(By.XPATH, '//h1[contains(text(), "Welcome")]')
assert welcome_message.is_displayed(), "Login failed"

# Close the browser
driver.quit()
