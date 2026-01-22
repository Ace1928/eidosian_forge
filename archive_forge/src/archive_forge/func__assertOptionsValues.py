from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def _assertOptionsValues(self, options_selector, values):
    from selenium.webdriver.common.by import By
    if values:
        options = self.selenium.find_elements(By.CSS_SELECTOR, options_selector)
        actual_values = []
        for option in options:
            actual_values.append(option.get_attribute('value'))
        self.assertEqual(values, actual_values)
    else:
        with self.disable_implicit_wait():
            self.wait_until(lambda driver: not driver.find_elements(By.CSS_SELECTOR, options_selector))