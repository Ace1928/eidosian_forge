from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import migrationutils
@mock.patch.object(migrationutils.MigrationUtils, '_get_planned_vm')
def test_planned_vm_exists(self, mock_get_planned_vm):
    mock_get_planned_vm.return_value = None
    result = self._migrationutils.planned_vm_exists(mock.sentinel.vm_name)
    self.assertFalse(result)
    mock_get_planned_vm.assert_called_once_with(mock.sentinel.vm_name)