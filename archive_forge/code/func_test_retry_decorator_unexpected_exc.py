from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_retry_decorator_unexpected_exc(self):
    self._test_retry_decorator_no_retry(expected_exceptions=(IOError, AttributeError))