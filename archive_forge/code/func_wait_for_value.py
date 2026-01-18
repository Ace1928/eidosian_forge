from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def wait_for_value(self, css_selector, text, timeout=10):
    """
        Block until the value is found in the CSS selector.
        """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    self.wait_until(ec.text_to_be_present_in_element_value((By.CSS_SELECTOR, css_selector), text), timeout)