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
@mock.patch('glob.glob')
def test_set_portals_controllers_do_nothing(self, mock_glob):
    """Do nothing if all protals already have the controller name."""
    self.target.portals[0].controller = 'nvme0'
    self.target.portals[1].controller = 'nvme1'
    self.target.set_portals_controllers()
    mock_glob.assert_not_called()