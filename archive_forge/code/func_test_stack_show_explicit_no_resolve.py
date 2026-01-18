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
def test_stack_show_explicit_no_resolve(self):
    arglist = ['--no-resolve-outputs', '--format', self.format, 'my_stack']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.stack_client.get.assert_called_with(**{'stack_id': 'my_stack', 'resolve_outputs': False})