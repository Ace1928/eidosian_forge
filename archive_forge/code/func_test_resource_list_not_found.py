import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_list_not_found(self):
    arglist = ['bad_stack']
    self.resource_client.list.side_effect = heat_exc.HTTPNotFound
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)