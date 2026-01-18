import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_metadata_error(self):
    arglist = ['my_stack', 'my_resource']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.resource_client.metadata.side_effect = heat_exc.HTTPNotFound
    error = self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
    self.assertEqual('Stack my_stack or resource my_resource not found.', str(error))