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
@mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('ADOPT_FAILED', 'Stack my_stack ADOPT_FAILED'))
def test_stack_adopt_wait_fail(self, mock_poll):
    arglist = ['my_stack', '--adopt-file', self.adopt_file, '--wait']
    self.stack_client.get.return_value = stacks.Stack(None, {'stack_status': 'ADOPT_FAILED'})
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)