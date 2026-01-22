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
@ddt.ddt
class UtilityMethodsTestCase(test_base.TestCase):

    @mock.patch.object(nvmeof, 'sysfs_property', return_value='live')
    def test_ctrl_property(self, mock_sysfs):
        """Controller properties just read from nvme fabrics in sysfs."""
        res = nvmeof.ctrl_property('state', 'nvme0')
        self.assertEqual('live', res)
        mock_sysfs.assert_called_once_with('state', '/sys/class/nvme-fabrics/ctl/nvme0')

    @mock.patch.object(nvmeof, 'sysfs_property', return_value='uuid_value')
    def test_blk_property(self, mock_sysfs):
        """Block properties just read from block devices in sysfs."""
        res = nvmeof.blk_property('uuid', 'nvme0n1')
        self.assertEqual('uuid_value', res)
        mock_sysfs.assert_called_once_with('uuid', '/sys/class/block/nvme0n1')

    @mock.patch.object(builtins, 'open')
    def test_sysfs_property(self, mock_open):
        """Method is basically an open an read method."""
        mock_read = mock_open.return_value.__enter__.return_value.read
        mock_read.return_value = ' uuid '
        res = nvmeof.sysfs_property('uuid', '/sys/class/block/nvme0n1')
        self.assertEqual('uuid', res)
        mock_open.assert_called_once_with('/sys/class/block/nvme0n1/uuid', 'r')
        mock_read.assert_called_once_with()

    @mock.patch.object(builtins, 'open', side_effect=FileNotFoundError)
    def test_sysfs_property_not_found(self, mock_open):
        """Failure to open file returns None."""
        mock_read = mock_open.return_value.__enter__.return_value.read
        res = nvmeof.sysfs_property('uuid', '/sys/class/block/nvme0n1')
        self.assertIsNone(res)
        mock_open.assert_called_once_with('/sys/class/block/nvme0n1/uuid', 'r')
        mock_read.assert_not_called()

    @mock.patch.object(builtins, 'open')
    def test_sysfs_property_ioerror(self, mock_open):
        """Failure to read file returns None."""
        mock_read = mock_open.return_value.__enter__.return_value.read
        mock_read.side_effect = IOError
        res = nvmeof.sysfs_property('uuid', '/sys/class/block/nvme0n1')
        self.assertIsNone(res)
        mock_open.assert_called_once_with('/sys/class/block/nvme0n1/uuid', 'r')
        mock_read.assert_called_once_with()

    @ddt.data('/dev/nvme0n10', '/sys/class/block/nvme0c1n10', '/sys/class/nvme-fabrics/ctl/nvme1/nvme0c1n10')
    def test_nvme_basename(self, name):
        """ANA devices are transformed to the right name."""
        res = nvmeof.nvme_basename(name)
        self.assertEqual('nvme0n10', res)