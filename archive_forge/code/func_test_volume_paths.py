import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
@mock.patch.object(smbfs.WindowsSMBFSConnector, '_get_disk_path')
def test_volume_paths(self, mock_get_disk_path):
    expected_paths = [mock_get_disk_path.return_value]
    volume_paths = self._connector.get_volume_paths(mock.sentinel.conn_props)
    self.assertEqual(expected_paths, volume_paths)
    mock_get_disk_path.assert_called_once_with(mock.sentinel.conn_props)