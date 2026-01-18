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
@mock.patch.object(os.path, 'exists', return_value=False)
@mock.patch('os_brick.utils._time_sleep')
def test_wait_for_path_not_found(self, exists_mock, sleep_mock):
    path = '/dev/disk/by-id/dm-uuid-mpath-%s' % '1234567890'
    self.assertRaisesRegex(exception.VolumeDeviceNotFound, 'Volume device not found at %s' % path, self.linuxscsi.wait_for_path, path)