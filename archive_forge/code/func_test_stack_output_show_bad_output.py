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
def test_stack_output_show_bad_output(self):
    arglist = ['my_stack', 'output3']
    self.stack_client.output_show.side_effect = heat_exc.HTTPNotFound
    self.stack_client.get.side_effect = heat_exc.HTTPNotFound
    parsed_args = self.check_parser(self.cmd, arglist, [])
    error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('Stack my_stack or output output3 not found.', str(error))
    self.stack_client.output_show.assert_called_with('my_stack', 'output3')