import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@ddt.data('/dev/nvme0n10', '/sys/class/block/nvme0c1n10', '/sys/class/nvme-fabrics/ctl/nvme1/nvme0c1n10')
def test_nvme_basename(self, name):
    """ANA devices are transformed to the right name."""
    res = nvmeof.nvme_basename(name)
    self.assertEqual('nvme0n10', res)