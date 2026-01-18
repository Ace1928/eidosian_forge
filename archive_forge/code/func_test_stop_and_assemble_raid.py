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
@mock.patch.object(nvmeof.NVMeOFConnector, 'ks_readlink')
@mock.patch.object(nvmeof.NVMeOFConnector, 'get_md_name')
def test_stop_and_assemble_raid(self, mock_md_name, mock_readlink):
    mock_readlink.return_value = '/dev/md/mdalias'
    mock_md_name.return_value = 'mdalias'
    self.assertIsNone(self.connector.stop_and_assemble_raid(['/dev/sda'], '/dev/md/mdalias', False))
    mock_md_name.assert_called_with('sda')
    mock_readlink.assert_called_with('/dev/md/mdalias')