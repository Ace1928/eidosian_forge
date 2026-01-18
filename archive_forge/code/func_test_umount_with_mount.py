import sys
from unittest import mock
import fixtures
from oslo_concurrency import processutils
from oslo_config import cfg
from oslotest import base
from glance_store import exceptions
@mock.patch('os.rmdir')
@mock.patch('os.makedirs')
def test_umount_with_mount(self, mock_makedirs, mock_rmdir):
    self._sentinel_mount()
    self._sentinel_umount()
    mock_makedirs.assert_called_once()
    mock_rmdir.assert_called_once()
    processutils.execute.assert_has_calls(self._expected_sentinel_mount_calls() + self._expected_sentinel_umount_calls())