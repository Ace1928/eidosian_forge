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
def test_wait_for_provision_state_one_timeout(self, mock_get, mock_sleep):
    fake_waiting_node = self._fake_node_for_wait('deploying', target='active')
    fake_success_node = self._fake_node_for_wait('active')

    def side_effect(node_manager, node_ident, *args, **kwargs):
        if node_ident == 'node1':
            return fake_success_node
        else:
            return fake_waiting_node
    mock_get.side_effect = side_effect
    self.assertRaisesRegex(exc.StateTransitionTimeout, 'Node\\(s\\) node2', self.mgr.wait_for_provision_state, ['node1', 'node2'], 'active', timeout=0.001)