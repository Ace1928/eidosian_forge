import os
import sys
import time
import logging
import warnings
import percy
import requests
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
from dash.testing.wait import (
from dash.testing.dash_page import DashPageMixin
from dash.testing.errors import DashAppLoadingError, BrowserError, TestingTimeoutError
from dash.testing.consts import SELENIUM_GRID_DEFAULT
def _get_firefox(self):
    options = self._get_wd_options()
    options.set_capability('loggingPrefs', {'browser': 'SEVERE'})
    options.set_capability('marionette', True)
    options.set_preference('browser.download.dir', self.download_path)
    options.set_preference('browser.download.folderList', 2)
    options.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/octet-stream')
    return webdriver.Remote(command_executor=self._remote_url, options=options) if self._remote else webdriver.Firefox(options=options)