from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def wait_until_visible(self, css_selector, timeout=10):
    """
        Block until the element described by the CSS selector is visible.
        """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as ec
    self.wait_until(ec.visibility_of_element_located((By.CSS_SELECTOR, css_selector)), timeout)