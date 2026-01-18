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
@mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('UPDATE_FAILED', 'Stack my_stack UPDATE_FAILED'))
@mock.patch('heatclient.common.event_utils.get_events', return_value=[])
def test_stack_update_wait_fail(self, ge, mock_poll):
    arglist = ['my_stack', '-t', self.template_path, '--wait']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.stack_client.get.return_value = mock.MagicMock(stack_name='my_stack')
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)