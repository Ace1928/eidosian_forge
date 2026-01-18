from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
@mock.patch.object(diskutils, '_RESCAN_LOCK')
@mock.patch.object(diskutils.DiskUtils, '_rescan_disks')
def test_rescan_merge_requests(self, mock_rescan_helper, mock_rescan_lock):
    mock_rescan_lock.locked.side_effect = [False, True, True]
    self._diskutils.rescan_disks(merge_requests=True)
    self._diskutils.rescan_disks(merge_requests=True)
    self._diskutils.rescan_disks(merge_requests=False)
    exp_rescan_count = 2
    mock_rescan_helper.assert_has_calls([mock.call()] * exp_rescan_count)
    mock_rescan_lock.__enter__.assert_has_calls([mock.call()] * exp_rescan_count)