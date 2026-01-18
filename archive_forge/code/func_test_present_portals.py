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
@mock.patch.object(nvmeof.Portal, 'state', new_callable=mock.PropertyMock)
def test_present_portals(self, mock_state):
    """List with only live portals should be returned."""
    self.target.portals.extend(self.target.portals)
    mock_state.side_effect = (None, 'live', 'connecting', 'live')
    res = self.target.present_portals
    self.assertListEqual(self.target.portals[1:], res)