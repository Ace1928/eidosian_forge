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
@mock.patch('os.path.realpath', side_effect=['/dev/sda', '/dev/sdb'])
def test_remove_scsi_symlinks_no_links(self, realpath_mock, glob_mock, unlink_mock):
    paths = ['/dev/disk/by-id/scsi-wwid1', '/dev/disk/by-id/scsi-wwid2']
    glob_mock.return_value = paths
    self.linuxscsi._remove_scsi_symlinks(['sdd', 'sde'])
    glob_mock.assert_called_once_with('/dev/disk/by-id/scsi-*')
    realpath_mock.assert_has_calls([mock.call(g) for g in paths])
    unlink_mock.assert_not_called()