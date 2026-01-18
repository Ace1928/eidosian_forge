from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_drive_to_boot_source')
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
def test_set_boot_order_gen2(self, mock_modify_virtual_system, mock_drive_to_boot_source):
    fake_dev_order = ['fake_boot_source1', 'fake_boot_source2']
    mock_drive_to_boot_source.side_effect = fake_dev_order
    mock_vssd = self._lookup_vm()
    old_boot_order = tuple(['fake_boot_source2', 'fake_boot_source1', 'fake_boot_source_net'])
    expected_boot_order = tuple(['FAKE_BOOT_SOURCE1', 'FAKE_BOOT_SOURCE2', 'FAKE_BOOT_SOURCE_NET'])
    mock_vssd.BootSourceOrder = old_boot_order
    self._vmutils._set_boot_order_gen2(mock_vssd.name, fake_dev_order)
    mock_modify_virtual_system.assert_called_once_with(mock_vssd)
    self.assertEqual(expected_boot_order, mock_vssd.BootSourceOrder)