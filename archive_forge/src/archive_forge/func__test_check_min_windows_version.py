from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def _test_check_min_windows_version(self, version, expected):
    os = mock.MagicMock()
    os.Version = version
    self._hostutils._conn_cimv2.Win32_OperatingSystem.return_value = [os]
    hostutils.HostUtils._windows_version = None
    self.assertEqual(expected, self._hostutils.check_min_windows_version(6, 2))