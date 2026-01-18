import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_list_with_detail(self):
    arglist = ['my_stack', '--long']
    cols = copy.deepcopy(self.columns)
    cols.append('stack_name')
    out = copy.deepcopy(self.data)
    out.append('my_stack')
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=True, nested_depth=None)
    self.assertEqual(cols, columns)
    self.assertEqual(tuple(out), list(data)[0])