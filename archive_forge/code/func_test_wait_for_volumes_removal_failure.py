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
@mock.patch('os_brick.utils._time_sleep')
@mock.patch('os.path.exists', return_value=True)
def test_wait_for_volumes_removal_failure(self, exists_mock, sleep_mock):
    retries = 61
    names = ('sda', 'sdb')
    self.assertRaises(exception.VolumePathNotRemoved, self.linuxscsi.wait_for_volumes_removal, names)
    exists_mock.assert_has_calls([mock.call('/dev/' + name) for name in names] * retries)
    self.assertEqual(retries - 1, sleep_mock.call_count)