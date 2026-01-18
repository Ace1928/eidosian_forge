import copy
import tempfile
import time
from unittest import mock
import testtools
from testtools.matchers import HasLength
from ironicclient.common import utils as common_utils
from ironicclient import exc
from ironicclient.tests.unit import utils
from ironicclient.v1 import node
from ironicclient.v1 import volume_connector
from ironicclient.v1 import volume_target
@mock.patch.object(time, 'sleep', autospec=True)
@mock.patch.object(node.NodeManager, 'get', autospec=True)
def test_wait_for_provision_state_timeout(self, mock_get, mock_sleep):
    mock_get.return_value = self._fake_node_for_wait('deploying', target='active')
    self.assertRaises(exc.StateTransitionTimeout, self.mgr.wait_for_provision_state, 'node', 'active', timeout=0.001)