import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_list_no_resource_name(self):
    arglist = ['my_stack']
    resp = copy.deepcopy(self.response)
    del resp['resource_name']
    cols = copy.deepcopy(self.columns)
    cols[0] = 'logical_resource_id'
    out = copy.deepcopy(self.data)
    out[1] = '1234'
    self.resource_client.list.return_value = [v1_resources.Resource(None, resp)]
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.resource_client.list.assert_called_with('my_stack', filters={}, with_detail=False, nested_depth=None)
    self.assertEqual(cols, columns)