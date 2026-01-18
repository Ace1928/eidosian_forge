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
@mock.patch.object(executor.Executor, '_execute')
def test_run_nvme_cli(self, mock_execute):
    mock_execute.return_value = ('\n', '')
    cmd = 'dummy command'
    result = self.connector.run_nvme_cli(cmd)
    self.assertEqual(('\n', ''), result)