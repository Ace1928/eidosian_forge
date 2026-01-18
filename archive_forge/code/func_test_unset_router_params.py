from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_unset_router_params(self):
    arglist = ['--route', 'destination=192.168.101.1/24,gateway=172.24.4.3', self._testrouter.name]
    verifylist = [('routes', [{'destination': '192.168.101.1/24', 'gateway': '172.24.4.3'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'routes': [{'destination': '192.168.101.2/24', 'nexthop': '172.24.4.3'}]}
    self.network_client.update_router.assert_called_once_with(self._testrouter, **attrs)
    self.assertIsNone(result)