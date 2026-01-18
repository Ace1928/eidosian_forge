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
def test_process_lun_id_list(self):
    lun_list = [2, 255, 88, 370, 5, 256]
    result = self.linuxscsi.process_lun_id(lun_list)
    expected = [2, 255, 88, '0x0172000000000000', 5, '0x0100000000000000']
    self.assertEqual(expected, result)