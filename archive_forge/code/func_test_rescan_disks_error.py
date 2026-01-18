from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch('time.sleep')
def test_rescan_disks_error(self, mock_sleep):
    mock_rescan = self._get_mocked_wmi_rescan(return_value=1)
    expected_retry_count = 5
    self.assertRaises(exceptions.OSWinException, self._diskutils.rescan_disks)
    mock_rescan.assert_has_calls([mock.call()] * expected_retry_count)