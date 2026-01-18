from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource_type
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resource_types
def test_resourcetype_list(self):
    arglist = []
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, rows = self.cmd.take_action(parsed_args)
    self.mock_client.resource_types.list.assert_called_with(filters={}, with_description=False)
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_rows, rows)