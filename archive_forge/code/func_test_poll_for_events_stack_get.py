from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@mock.patch('heatclient.common.event_utils.get_events')
def test_poll_for_events_stack_get(self, ge):
    mock_client = mock.MagicMock()
    mock_client.stacks.get.return_value.stack_status = 'CREATE_FAILED'
    ge.side_effect = [[self._mock_stack_event('1', 'astack', 'CREATE_IN_PROGRESS'), self._mock_event('2', 'res_child1', 'CREATE_IN_PROGRESS'), self._mock_event('3', 'res_child2', 'CREATE_IN_PROGRESS'), self._mock_event('4', 'res_child3', 'CREATE_IN_PROGRESS')], [], []]
    stack_status, msg = event_utils.poll_for_events(mock_client, 'astack', action='CREATE', poll_period=0)
    self.assertEqual('CREATE_FAILED', stack_status)
    self.assertEqual('\n Stack astack CREATE_FAILED \n', msg)