import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
def test_read_mounts(self):
    mounts = 'device1 mnt_point1 ext4 rw,seclabel,relatime 0 0\n                    device2 mnt_point2 ext4 rw,seclabel,relatime 0 0'
    with mock.patch('os_brick.remotefs.remotefs.open', mock.mock_open(read_data=mounts)) as mock_open:
        client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/mnt')
        ret = client._read_mounts()
        mock_open.assert_called_once_with('/proc/mounts', 'r')
    self.assertEqual(ret, {'mnt_point1': 'device1', 'mnt_point2': 'device2'})