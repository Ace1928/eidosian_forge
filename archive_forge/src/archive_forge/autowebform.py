from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

# Create a new instance of the Chrome driver
driver = webdriver.Chrome()

# Navigate to the web form page
driver.get("https://example.com/form")

# Fill out the form fields
name_field = driver.find_element(By.NAME, "name")
name_field.send_keys("John Doe")

email_field = driver.find_element(By.NAME, "email")
email_field.send_keys("john@example.com")

country_select = Select(driver.find_element(By.NAME, "country"))
country_select.select_by_visible_text("United States")

# Submit the form
submit_button = driver.find_element(By.XPATH, '//button[@type="submit"]')
submit_button.click()

# Close the browser
driver.quit()
