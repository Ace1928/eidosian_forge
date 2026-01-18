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
@mock.patch('heatclient.common.event_utils.poll_for_events', return_value=('UPDATE_COMPLETE', 'Stack my_stack UPDATE_COMPLETE'))
@mock.patch('heatclient.common.event_utils.get_events', return_value=[])
def test_stack_update_wait(self, ge, mock_poll):
    arglist = ['my_stack', '-t', self.template_path, '--wait']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.stack_client.get.return_value = mock.MagicMock(stack_name='my_stack')
    self.cmd.take_action(parsed_args)
    self.stack_client.update.assert_called_with(**self.defaults)
    self.stack_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resolve_outputs': False})