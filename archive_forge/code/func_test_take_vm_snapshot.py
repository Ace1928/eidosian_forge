from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
@ddt.data(None, mock.sentinel.snap_name)
def test_take_vm_snapshot(self, snap_name, mock_modify_virtual_system):
    self._lookup_vm()
    mock_snap = mock.Mock(ElementName=mock.sentinel.default_snap_name)
    mock_svc = self._get_snapshot_service()
    mock_svc.CreateSnapshot.return_value = (self._FAKE_JOB_PATH, mock.MagicMock(), self._FAKE_RET_VAL)
    mock_job = self._vmutils._jobutils.check_ret_val.return_value
    mock_job.associators.return_value = [mock_snap]
    snap_path = self._vmutils.take_vm_snapshot(self._FAKE_VM_NAME, snap_name)
    self.assertEqual(mock_snap.path_.return_value, snap_path)
    mock_svc.CreateSnapshot.assert_called_with(AffectedSystem=self._FAKE_VM_PATH, SnapshotType=self._vmutils._SNAPSHOT_FULL)
    self._vmutils._jobutils.check_ret_val.assert_called_once_with(self._FAKE_RET_VAL, self._FAKE_JOB_PATH)
    mock_job.associators.assert_called_once_with(wmi_result_class=self._vmutils._VIRTUAL_SYSTEM_SETTING_DATA_CLASS, wmi_association_class=self._vmutils._AFFECTED_JOB_ELEMENT_CLASS)
    if snap_name:
        self.assertEqual(snap_name, mock_snap.ElementName)
        mock_modify_virtual_system.assert_called_once_with(mock_snap)
    else:
        self.assertEqual(mock.sentinel.default_snap_name, mock_snap.ElementName)
        mock_modify_virtual_system.assert_not_called()