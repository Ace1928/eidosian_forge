import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
def test_vzstorage_invalid_share(self, mock_read_mounts):
    client = remotefs.VZStorageRemoteFSClient('vzstorage', root_helper='true', vzstorage_mount_point_base='/mnt')
    self.assertRaises(exception.BrickException, client.mount, ':')