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
def test_stack_output_show_output(self):
    arglist = ['my_stack', 'output1']
    self.stack_client.output_show.return_value = {'output': self.outputs[0]}
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, outputs = self.cmd.take_action(parsed_args)
    self.stack_client.output_show.assert_called_with('my_stack', 'output1')
    self.assertEqual(('output_key', 'output_value'), columns)
    self.assertEqual(('output1', 'value1'), outputs)