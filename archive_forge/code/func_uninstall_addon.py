import base64
import logging
import os
import warnings
import zipfile
from contextlib import contextmanager
from io import BytesIO
from selenium.webdriver.common.driver_finder import DriverFinder
from selenium.webdriver.remote.webdriver import WebDriver as RemoteWebDriver
from .options import Options
from .remote_connection import FirefoxRemoteConnection
from .service import Service
def uninstall_addon(self, identifier) -> None:
    """Uninstalls Firefox addon using its identifier.

        :Usage:
            ::

                driver.uninstall_addon('addon@foo.com')
        """
    self.execute('UNINSTALL_ADDON', {'id': identifier})