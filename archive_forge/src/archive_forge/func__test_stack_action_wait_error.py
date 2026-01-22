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
def _test_stack_action_wait_error(self, ge, mock_poll):
    arglist = ['my_stack', '--wait']
    mock_poll.return_value = ('%s_FAILED' % self.action_name, 'Error waiting for status from stack my_stack')
    parsed_args = self.check_parser(self.cmd, arglist, [])
    error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('Error waiting for status from stack my_stack', str(error))