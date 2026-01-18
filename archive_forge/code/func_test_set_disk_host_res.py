from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
def test_set_disk_host_res(self, mock_get_wmi_obj):
    mock_diskdrive = mock_get_wmi_obj.return_value
    self._vmutils.set_disk_host_res(self._FAKE_RES_PATH, self._FAKE_MOUNTED_DISK_PATH)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_diskdrive)
    mock_get_wmi_obj.assert_called_once_with(self._FAKE_RES_PATH, True)
    self.assertEqual(mock_diskdrive.HostResource, [self._FAKE_MOUNTED_DISK_PATH])