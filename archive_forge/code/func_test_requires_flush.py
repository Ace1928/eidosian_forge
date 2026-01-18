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
@ddt.data(('/dev/sda', '/dev/sda', False, True, None), ('/dev/sda', '/dev/sda', True, True, None), ('/dev/sda', '', True, False, None), ('/dev/link_sda', '/dev/disk/by-path/pci-XYZ', False, True, ('/dev/sda', '/dev/mapper/crypt-pci-XYZ')), ('/dev/link_sda', '/dev/link_sdb', False, False, ('/dev/sda', '/dev/sdb')), ('/dev/link_sda', '/dev/link2_sda', False, True, ('/dev/sda', '/dev/sda')))
@ddt.unpack
def test_requires_flush(self, path, path_used, was_multipath, expected, real_paths):
    with mock.patch('os.path.realpath', side_effect=real_paths) as mocked:
        self.assertEqual(expected, self.linuxscsi.requires_flush(path, path_used, was_multipath))
        if real_paths:
            mocked.assert_has_calls([mock.call(path), mock.call(path_used)])