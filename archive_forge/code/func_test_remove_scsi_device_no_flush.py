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
def test_remove_scsi_device_no_flush(self, exists_mock):
    self.linuxscsi.remove_scsi_device('/dev/sdc')
    expected_commands = []
    self.assertEqual(expected_commands, self.cmds)
    exists_mock.return_value = True
    self.linuxscsi.remove_scsi_device('/dev/sdc', flush=False)
    expected_commands = ['tee -a /sys/block/sdc/device/delete']
    self.assertEqual(expected_commands, self.cmds)