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
@mock.patch.object(linuxscsi.priv_rootwrap, 'unlink_root')
@mock.patch('glob.glob')
@mock.patch('os.path.realpath', side_effect=['/dev/sda', '/dev/sdb', '/dev/sdc'])
def test_remove_scsi_symlinks(self, realpath_mock, glob_mock, unlink_mock):
    paths = ['/dev/disk/by-id/scsi-wwid1', '/dev/disk/by-id/scsi-wwid2', '/dev/disk/by-id/scsi-wwid3']
    glob_mock.return_value = paths
    self.linuxscsi._remove_scsi_symlinks(['sdb', 'sdc', 'sdd'])
    glob_mock.assert_called_once_with('/dev/disk/by-id/scsi-*')
    realpath_mock.assert_has_calls([mock.call(g) for g in paths])
    unlink_mock.assert_called_once_with(*paths[1:], no_errors=True)