from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def wait_page_ready(self, timeout=10):
    """
        Block until the  page is ready.
        """
    self.wait_until(lambda driver: driver.execute_script('return document.readyState;') == 'complete', timeout)