import os
import time
import logging
import shutil
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

# Constants
BASE_URL = "https://www.freepik.com/search?format=search&last_filter=type&last_value=vector&query=kids+coloring&selection=1&type=vector"
DOWNLOAD_DIR = "coloring_images"
REQUEST_DELAY = 2  # seconds
FIREFOX_PROFILE_DIR = "firefox_profile"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Log environment details
logging.info(f"Operating System: {os.name}")
logging.info(f"Python Version: {os.sys.version}")
logging.info(f"Requests Version: {requests.__version__}")
logging.info(f"Selenium Version: {webdriver.__version__}")

# Create download directory if it doesn't exist
try:
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
        logging.info(f"Created download directory at {DOWNLOAD_DIR}")
except Exception as e:
    logging.error(f"Failed to create download directory: {e}")
    raise

# Create Firefox profile directory if it doesn't exist
try:
    if not os.path.exists(FIREFOX_PROFILE_DIR):
        os.makedirs(FIREFOX_PROFILE_DIR)
        logging.info(f"Created Firefox profile directory at {FIREFOX_PROFILE_DIR}")
except Exception as e:
    logging.error(f"Failed to create Firefox profile directory: {e}")
    raise

# Ensure the correct GeckoDriver and Firefox version are installed
try:
    geckodriver_path = GeckoDriverManager().install()
    logging.info(f"GeckoDriver installed at {geckodriver_path}")
except Exception as e:
    logging.error(f"Failed to install GeckoDriver: {e}")
    raise

# Set up Firefox options
try:
    options = Options()
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", os.path.abspath(DOWNLOAD_DIR))
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/zip")
    options.add_argument(f"--profile={FIREFOX_PROFILE_DIR}")
    logging.info("Firefox options set with the required preferences.")
except Exception as e:
    logging.error(f"Failed to set Firefox options: {e}")
    raise

# Initialize the WebDriver
try:
    driver = webdriver.Firefox(service=Service(geckodriver_path), options=options)
    logging.info("WebDriver initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize WebDriver: {e}")
    raise


def get_image_links(page_url):
    try:
        driver.get(page_url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "showcase__image"))
        )
        image_elements = driver.find_elements(By.CLASS_NAME, "showcase__image")
        image_links = [element.get_attribute("src") for element in image_elements]
        logging.info(f"Found {len(image_links)} image links on {page_url}")
        return image_links
    except Exception as e:
        logging.error(f"Failed to get image links from {page_url}: {e}")
        return []


def download_image(url, folder, image_num):
    try:
        driver.get(url)
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
    try:
        page_num = 1
        while True:
            page_url = f"{BASE_URL}&page={page_num}"
            logging.info(f"Fetching image links from {page_url}")
            image_links = get_image_links(page_url)

            if not image_links:
                logging.info("No more images found. Exiting.")
                break

            for i, link in enumerate(image_links):
                download_image(
                    link, DOWNLOAD_DIR, (page_num - 1) * len(image_links) + i + 1
                )

            page_num += 1
            time.sleep(REQUEST_DELAY)  # Respectful delay between requests

    except Exception as e:
        logging.error(f"An error occurred in the main loop: {e}")
    finally:
        driver.quit()
        logging.info("Finished downloading all images and closed the WebDriver.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
        raise
