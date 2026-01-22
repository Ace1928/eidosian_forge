from contextlib import contextmanager
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.test import modify_settings
from django.test.selenium import SeleniumTestCase
from django.utils.deprecation import MiddlewareMixin
from django.utils.translation import gettext as _
def assertSelectOptions(self, selector, values):
    """
        Assert that the <SELECT> widget identified by `selector` has the
        options with the given `values`.
        """
    self._assertOptionsValues('%s > option' % selector, values)