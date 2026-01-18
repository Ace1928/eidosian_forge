from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import privileged
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('oslo_concurrency.processutils.execute', side_effect=OSError(42, 'mock error'))
def test_oserror_raise(self, mock_putils_exec):
    self.assertRaises(putils.ProcessExecutionError, priv_rootwrap.execute, 'foo')