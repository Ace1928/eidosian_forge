import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_update_planned_vm_disk_resources(self, mock_get_elem_associated_class):
    self._prepare_vm_mocks(self._RESOURCE_TYPE_DISK, self._RESOURCE_SUB_TYPE_DISK, mock_get_elem_associated_class)
    mock_vm = mock.Mock(Name='fake_name')
    sasd = mock_get_elem_associated_class.return_value[0]
    mock_vsmsvc = self._conn.Msvm_VirtualSystemManagementService()[0]
    self.liveutils._update_planned_vm_disk_resources(self._conn, mock_vm, mock.sentinel.FAKE_VM_NAME, {sasd.path.return_value.RelPath: mock.sentinel.FAKE_RASD_PATH})
    mock_vsmsvc.ModifyResourceSettings.assert_called_once_with(ResourceSettings=[sasd.GetText_.return_value])
    mock_get_elem_associated_class.assert_called_once_with(self._conn, self.liveutils._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_uuid=mock_vm.Name)