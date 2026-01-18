from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_mounted_disk_resource_from_path')
def test_set_disk_qos_specs_missing_values(self, mock_get_disk_resource):
    self._vmutils.set_disk_qos_specs(mock.sentinel.disk_path)
    self.assertFalse(mock_get_disk_resource.called)