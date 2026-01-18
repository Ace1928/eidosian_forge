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
def visit_and_snapshot(self, resource_path, hook_id, wait_for_callbacks=True, convert_canvases=False, assert_check=True, stay_on_page=False, widths=None):
    try:
        path = resource_path.lstrip('/')
        if path != resource_path:
            logger.warning("we stripped the left '/' in resource_path")
        self.driver.get(f'{self.server_url.rstrip('/')}/{path}')
        self.wait_for_element_by_id(hook_id)
        self.percy_snapshot(path, wait_for_callbacks=wait_for_callbacks, convert_canvases=convert_canvases, widths=widths)
        if assert_check:
            assert not self.find_elements('div.dash-debug-alert'), 'devtools should not raise an error alert'
        if not stay_on_page:
            self.driver.back()
    except WebDriverException as e:
        logger.exception('snapshot at resource %s error', path)
        raise e