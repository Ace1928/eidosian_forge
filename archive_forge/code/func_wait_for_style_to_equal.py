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
def wait_for_style_to_equal(self, selector, style, val, timeout=None):
    """Explicit wait until the element's style has expected `value` timeout
        if not set, equals to the fixture's `wait_timeout` shortcut to
        `WebDriverWait` with customized `style_to_equal` condition."""
    return self._wait_for(method=style_to_equal(selector, style, val), timeout=timeout, msg=f'style val => {style} {val} not found within {timeout or self._wait_timeout}s')