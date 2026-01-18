from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data({'vnuma_enabled': mock.sentinel.vnuma_enabled}, {'configuration_root_dir': mock.sentinel.configuration_root_dir}, {'host_shutdown_action': mock.sentinel.shutdown_action}, {'chassis_asset_tag': mock.sentinel.chassis_asset_tag2}, {})
@ddt.unpack
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
@mock.patch.object(vmutils.VMUtils, '_set_vm_vcpus')
@mock.patch.object(vmutils.VMUtils, '_set_vm_memory')
@mock.patch.object(vmutils.VMUtils, '_set_vm_snapshot_type')
@mock.patch.object(vmutils.VMUtils, '_lookup_vm_check')
def test_update_vm(self, mock_lookup_vm_check, mock_set_vm_snap_type, mock_set_mem, mock_set_vcpus, mock_modify_virtual_system, host_shutdown_action=None, configuration_root_dir=None, vnuma_enabled=None, chassis_asset_tag=None):
    mock_vmsettings = mock_lookup_vm_check.return_value
    self._vmutils.update_vm(mock.sentinel.vm_name, mock.sentinel.memory_mb, mock.sentinel.memory_per_numa, mock.sentinel.vcpus_num, mock.sentinel.vcpus_per_numa, mock.sentinel.limit_cpu_features, mock.sentinel.dynamic_mem_ratio, configuration_root_dir, host_shutdown_action=host_shutdown_action, vnuma_enabled=vnuma_enabled, snapshot_type=mock.sentinel.snap_type, chassis_asset_tag=chassis_asset_tag)
    mock_lookup_vm_check.assert_called_once_with(mock.sentinel.vm_name, for_update=True)
    mock_set_mem.assert_called_once_with(mock_vmsettings, mock.sentinel.memory_mb, mock.sentinel.memory_per_numa, mock.sentinel.dynamic_mem_ratio)
    mock_set_vcpus.assert_called_once_with(mock_vmsettings, mock.sentinel.vcpus_num, mock.sentinel.vcpus_per_numa, mock.sentinel.limit_cpu_features)
    if configuration_root_dir:
        self.assertEqual(configuration_root_dir, mock_vmsettings.ConfigurationDataRoot)
        self.assertEqual(configuration_root_dir, mock_vmsettings.LogDataRoot)
        self.assertEqual(configuration_root_dir, mock_vmsettings.SnapshotDataRoot)
        self.assertEqual(configuration_root_dir, mock_vmsettings.SuspendDataRoot)
        self.assertEqual(configuration_root_dir, mock_vmsettings.SwapFileDataRoot)
    if host_shutdown_action:
        self.assertEqual(host_shutdown_action, mock_vmsettings.AutomaticShutdownAction)
    if vnuma_enabled:
        self.assertEqual(vnuma_enabled, mock_vmsettings.VirtualNumaEnabled)
    if chassis_asset_tag:
        self.assertEqual(chassis_asset_tag, mock_vmsettings.ChassisAssetTag)
    mock_set_vm_snap_type.assert_called_once_with(mock_vmsettings, mock.sentinel.snap_type)
    mock_modify_virtual_system.assert_called_once_with(mock_vmsettings)