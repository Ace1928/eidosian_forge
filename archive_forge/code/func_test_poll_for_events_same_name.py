from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
@mock.patch('heatclient.common.event_utils.get_events')
def test_poll_for_events_same_name(self, ge):
    ge.side_effect = [[self._mock_stack_event('1', 'mything', 'CREATE_IN_PROGRESS'), self._mock_event('2', 'res_child1', 'CREATE_IN_PROGRESS'), self._mock_event('3', 'mything', 'CREATE_IN_PROGRESS')], [self._mock_event('4', 'mything', 'CREATE_COMPLETE')], [self._mock_event('5', 'res_child1', 'CREATE_COMPLETE'), self._mock_stack_event('6', 'mything', 'CREATE_COMPLETE')]]
    stack_status, msg = event_utils.poll_for_events(None, 'mything', action='CREATE', poll_period=0)
    self.assertEqual('CREATE_COMPLETE', stack_status)
    self.assertEqual('\n Stack mything CREATE_COMPLETE \n', msg)
    ge.assert_has_calls([mock.call(None, stack_id='mything', nested_depth=0, event_args={'sort_dir': 'asc', 'marker': None}), mock.call(None, stack_id='mything', nested_depth=0, event_args={'sort_dir': 'asc', 'marker': '3'}), mock.call(None, stack_id='mything', nested_depth=0, event_args={'sort_dir': 'asc', 'marker': '4'})])