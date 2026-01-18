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
def test_process_lun_id_single_var_return(self):
    lun_id = 13
    result = self.linuxscsi.process_lun_id(lun_id)
    expected = 13
    self.assertEqual(expected, result)