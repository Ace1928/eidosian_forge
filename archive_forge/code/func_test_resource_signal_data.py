import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import resource
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import resources as v1_resources
def test_resource_signal_data(self):
    arglist = ['my_stack', 'my_resource', '--data', '{"message":"Content"}']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.resource_client.signal.assert_called_with(**{'data': {'message': 'Content'}, 'stack_id': 'my_stack', 'resource_name': 'my_resource'})