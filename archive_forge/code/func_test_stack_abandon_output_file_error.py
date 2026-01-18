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
@mock.patch('heatclient.osc.v1.stack.open', create=True, side_effect=IOError)
def test_stack_abandon_output_file_error(self, mock_open):
    arglist = ['my_stack', '--output-file', 'file.json']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)