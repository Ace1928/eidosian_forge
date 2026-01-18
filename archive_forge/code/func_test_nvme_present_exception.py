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
@ddt.data(OSError(2, 'FileNotFoundError'), Exception())
@mock.patch('os_brick.initiator.connectors.nvmeof.LOG')
@mock.patch.object(priv_rootwrap, 'custom_execute', autospec=True)
def test_nvme_present_exception(self, exc, mock_execute, mock_log):
    mock_execute.side_effect = exc
    nvme_present = self.connector.nvme_present()
    log = mock_log.debug if isinstance(exc, OSError) else mock_log.warning
    log.assert_called_once()
    self.assertFalse(nvme_present)