import os
import time
import logging
import shutil
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Constants
BASE_URL = "https://www.freepik.com/search?format=search&last_filter=type&last_value=vector&query=kids+coloring&selection=1&type=vector"
DOWNLOAD_DIR = "coloring_images"
REQUEST_DELAY = 2  # seconds
CHROME_DRIVER_PATH = "/home/lloyd/Downloads/chromedriver.exe"
CHROME_DRIVER_URL = (
    "https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_win32.zip"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Create download directory if it doesn't exist
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)

# Download ChromeDriver if it doesn't exist
if not os.path.exists(CHROME_DRIVER_PATH):
    logging.info("Downloading ChromeDriver...")
    response = requests.get(CHROME_DRIVER_URL, stream=True)
    with open("chromedriver.zip", "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    shutil.unpack_archive("chromedriver.zip", extract_dir=".")
    os.remove("chromedriver.zip")
    logging.info("ChromeDriver downloaded and extracted.")

# Configure Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_experimental_option(
    "prefs",
    {
        "download.default_directory": os.path.abspath(DOWNLOAD_DIR),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    },
)
service = Service(CHROME_DRIVER_PATH)
driver = webdriver.Chrome(service=service, options=chrome_options)


def get_image_links(page_url):
    driver.get(page_url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "showcase__image"))
    )
    image_elements = driver.find_elements(By.CLASS_NAME, "showcase__image")
    image_links = [element.get_attribute("src") for element in image_elements]
    return image_links


def download_image(url, folder, image_num):
    driver.get(url)
    try:
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "btn--download"))
        )
        download_button.click()
        free_download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "btn--free"))
        )
        free_download_button.click()
        logging.info(f"Initiated download for {url}")
        time.sleep(REQUEST_DELAY)  # Wait for the download to complete

        # Move the downloaded file to the specified folder
        download_path = os.path.join(DOWNLOAD_DIR, f"image_{image_num}.zip")
        while not os.path.exists(download_path):
            time.sleep(1)  # Wait until the file is downloaded
        shutil.move(download_path, os.path.join(folder, f"image_{image_num}.zip"))
        logging.info(f"Downloaded and moved {url} to {folder}")

    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")


def main():
    page_num = 1
    while True:
        page_url = f"{BASE_URL}&page={page_num}"
        logging.info(f"Fetching image links from {page_url}")
        image_links = get_image_links(page_url)
        if not image_links:
            logging.info("No more images found, exiting.")
            break  # No more images found, exit the loop
        for image_num, link in enumerate(image_links, start=1):
            download_image(link, DOWNLOAD_DIR, image_num)
            time.sleep(REQUEST_DELAY)
        page_num += 1


if __name__ == "__main__":
    try:
        main()
    finally:
        driver.quit()
        logging.info("WebDriver closed.")
