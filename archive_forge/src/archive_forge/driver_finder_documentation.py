import logging
from pathlib import Path
from selenium.common.exceptions import NoSuchDriverException
from selenium.webdriver.common.options import BaseOptions
from selenium.webdriver.common.selenium_manager import SeleniumManager
from selenium.webdriver.common.service import Service
Utility to find if a given file is present and executable.

    This implementation is still in beta, and may change.
    