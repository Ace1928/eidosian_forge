from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
def test_retry_decorator_time_exceeded(self):
    self._test_retry_decorator_exceeded(mock_time_side_eff=[0, 1, 4], timeout=3, expected_try_count=1)