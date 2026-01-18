from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@ddt.data(0, [0], (0,))
@mock.patch('time.sleep')
def test_rescan_disks(self, return_value, mock_sleep):
    mock_rescan = self._get_mocked_wmi_rescan(return_value)
    self._diskutils.rescan_disks()
    mock_rescan.assert_called_once_with()