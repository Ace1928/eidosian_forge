from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_set_disk_qos_specs(self, mock_get_disk_resource):
    mock_disk = mock_get_disk_resource.return_value
    self._vmutils.set_disk_qos_specs(mock.sentinel.disk_path, max_iops=mock.sentinel.max_iops, min_iops=mock.sentinel.min_iops)
    mock_get_disk_resource.assert_called_once_with(mock.sentinel.disk_path, is_physical=False)
    self.assertEqual(mock.sentinel.max_iops, mock_disk.IOPSLimit)
    self.assertEqual(mock.sentinel.min_iops, mock_disk.IOPSReservation)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_disk)