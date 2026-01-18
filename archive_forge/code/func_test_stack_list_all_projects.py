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
def test_stack_list_all_projects(self):
    self.stack_client.list.return_value = [stacks.Stack(None, self.data_with_project)]
    kwargs = copy.deepcopy(self.defaults)
    kwargs['global_tenant'] = True
    cols = copy.deepcopy(self.columns)
    cols.insert(2, 'Project')
    arglist = ['--all-projects']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.stack_client.list.assert_called_with(**kwargs)
    self.assertEqual(cols, columns)