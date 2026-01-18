import copy
import io
from unittest import mock
from osc_lib import exceptions as exc
from osc_lib import utils
import testscenarios
import yaml
from heatclient.common import template_format
from heatclient import exc as heat_exc
from heatclient.osc.v1 import stack
from heatclient.tests import inline_templates
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import events
from heatclient.v1 import resources
from heatclient.v1 import stacks
@mock.patch('heatclient.common.event_utils.poll_for_events')
@mock.patch('heatclient.common.event_utils.get_events', return_value=[])
def test_stack_delete_wait_fail(self, mock_get_event, mock_poll):
    mock_poll.side_effect = [['DELETE_COMPLETE', 'Stack my_stack DELETE_COMPLETE'], ['DELETE_FAILED', 'Stack my_stack DELETE_FAILED'], ['DELETE_COMPLETE', 'Stack my_stack DELETE_COMPLETE']]
    arglist = ['stack1', 'stack2', 'stack3', '--wait']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.stack_client.delete.assert_any_call('stack1')
    self.stack_client.delete.assert_any_call('stack2')
    self.stack_client.delete.assert_any_call('stack3')
    self.assertEqual('Unable to delete 1 of the 3 stacks.', str(error))