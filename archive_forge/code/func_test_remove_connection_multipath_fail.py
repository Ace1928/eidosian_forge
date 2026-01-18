import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch.object(linuxscsi.LinuxSCSI, '_remove_scsi_symlinks')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_del_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'is_multipath_running', return_value=True)
@mock.patch.object(linuxscsi.LinuxSCSI, 'flush_multipath_device', side_effect=Exception)
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_dm_name')
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_sysfs_multipath_dm')
@mock.patch.object(linuxscsi.LinuxSCSI, 'wait_for_volumes_removal')
@mock.patch.object(linuxscsi.LinuxSCSI, 'remove_scsi_device')
def test_remove_connection_multipath_fail(self, remove_mock, wait_mock, find_dm_mock, get_dm_name_mock, flush_mp_mock, is_mp_running_mock, mp_del_path_mock, remove_link_mock):
    flush_mp_mock.side_effect = exception.ExceptionChainer
    devices_names = ('sda', 'sdb')
    exc = exception.ExceptionChainer()
    self.assertRaises(exception.ExceptionChainer, self.linuxscsi.remove_connection, devices_names, force=False, exc=exc)
    find_dm_mock.assert_called_once_with(devices_names)
    get_dm_name_mock.assert_called_once_with(find_dm_mock.return_value)
    flush_mp_mock.assert_called_once_with(get_dm_name_mock.return_value)
    is_mp_running_mock.assert_not_called()
    mp_del_path_mock.assert_not_called()
    remove_mock.assert_not_called()
    wait_mock.assert_not_called()
    remove_link_mock.assert_not_called()
    self.assertTrue(bool(exc))