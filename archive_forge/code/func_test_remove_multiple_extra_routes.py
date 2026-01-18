from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_remove_multiple_extra_routes(self):
    arglist = [self._router.id, '--route', 'destination=dst1,gateway=gw1', '--route', 'destination=dst2,gateway=gw2']
    verifylist = [('router', self._router.id), ('routes', [{'destination': 'dst1', 'gateway': 'gw1'}, {'destination': 'dst2', 'gateway': 'gw2'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.remove_extra_routes_from_router.assert_called_with(self._router, body={'router': {'routes': [{'destination': 'dst1', 'nexthop': 'gw1'}, {'destination': 'dst2', 'nexthop': 'gw2'}]}})
    self.assertEqual(2, len(result))