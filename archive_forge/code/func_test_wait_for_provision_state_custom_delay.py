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
@mock.patch.object(node.NodeManager, 'get', autospec=True)
def test_wait_for_provision_state_custom_delay(self, mock_get):
    mock_get.side_effect = [self._fake_node_for_wait('deploying', target='active'), self._fake_node_for_wait('active')]
    delay_mock = mock.Mock()
    self.mgr.wait_for_provision_state('node', 'active', poll_delay_function=delay_mock)
    mock_get.assert_called_with(self.mgr, 'node', os_ironic_api_version=None, global_request_id=None)
    self.assertEqual(2, mock_get.call_count)
    delay_mock.assert_called_with(node._DEFAULT_POLL_INTERVAL)
    self.assertEqual(1, delay_mock.call_count)