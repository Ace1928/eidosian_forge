from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def _test_get_wt_snap(self, snap_found=True, fail_if_not_found=False):
    mock_wt_snap = mock.Mock()
    mock_wt_snap_cls = self._tgutils._conn_wmi.WT_Snapshot
    mock_wt_snap_cls.return_value = [mock_wt_snap] if snap_found else []
    if not snap_found and fail_if_not_found:
        self.assertRaises(exceptions.ISCSITargetException, self._tgutils._get_wt_snapshot, mock.sentinel.snap_description, fail_if_not_found=fail_if_not_found)
    else:
        wt_snap = self._tgutils._get_wt_snapshot(mock.sentinel.snap_description, fail_if_not_found=fail_if_not_found)
        expected_wt_snap = mock_wt_snap if snap_found else None
        self.assertEqual(expected_wt_snap, wt_snap)
    mock_wt_snap_cls.assert_called_once_with(Description=mock.sentinel.snap_description)