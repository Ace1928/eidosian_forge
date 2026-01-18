from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_unique_result(self):
    result = self.utils._unique_result([mock.sentinel.obj], mock.sentinel.resource_name)
    self.assertEqual(mock.sentinel.obj, result)