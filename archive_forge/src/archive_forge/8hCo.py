import os
import time
import logging
import shutil
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager

# Constants
BASE_URL = "https://www.freepik.com/search?format=search&last_filter=type&last_value=vector&query=kids+coloring&selection=1&type=vector"
DOWNLOAD_DIR = "coloring_images"
REQUEST_DELAY = 2  # seconds
FIREFOX_PROFILE_DIR = "firefox_profile"
MARIONETTE_PORT = 2828

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


def setup_driver():
    """
    Set up the Firefox WebDriver with a custom profile and options.

    Returns:
        WebDriver: Configured Firefox WebDriver instance.
    """
    options = Options()
    options.headless = False  # Set to True if you want to run in headless mode

    # Set up the Firefox profile
    profile = webdriver.FirefoxProfile()
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.download.dir", os.path.abspath(DOWNLOAD_DIR))
    profile.set_preference(
        "browser.helperApps.neverAsk.saveToDisk", "image/png,image/jpeg,image/jpg"
    )

    # Initialize the WebDriver
    service = Service(GeckoDriverManager().install())
    driver = webdriver.Firefox(
        service=service, options=options, firefox_profile=profile
    )
    return driver


# Initialize the WebDriver
driver = setup_driver()


def get_image_links(page_url: str) -> list:
    """
    Fetch image links from the given page URL.

    Args:
        page_url (str): URL of the page to fetch image links from.

    Returns:
        list: List of image URLs.
    """
    driver.get(page_url)
    try:
        # Wait for the images to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "img"))
        )
        images = driver.find_elements(By.CSS_SELECTOR, "img")
        image_links = [img.get_attribute("src") for img in images]
        logging.info(f"Found {len(image_links)} image links on {page_url}")
        return image_links
    except Exception as e:
        logging.error(f"Error fetching image links: {e}")
        return []


def download_image(url: str, folder: str, image_num: int) -> None:
    """
    Downloads an image from the given URL and saves it to the specified folder.

    Args:
        url (str): The URL of the image to download.
        folder (str): The folder to save the downloaded image.
        image_num (int): The image number for naming the downloaded file.

    Returns:
        None
    """
    try:
        driver.get(url)
        download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn--download"))
        )
        ActionChains(driver).move_to_element(download_button).click().perform()
        free_download_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn--free"))
        )
        ActionChains(driver).move_to_element(free_download_button).click().perform()
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


def main() -> None:
    """
    Main function to orchestrate the downloading of images from multiple pages.

    Returns:
        None
    """
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
