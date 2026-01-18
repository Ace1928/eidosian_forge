from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_route(self):
    arglist = [self._router.name, '--route', 'destination=10.20.30.0/24,gateway=10.20.30.1']
    verifylist = [('router', self._router.name), ('routes', [{'destination': '10.20.30.0/24', 'gateway': '10.20.30.1'}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    routes = [{'destination': '10.20.30.0/24', 'nexthop': '10.20.30.1'}]
    attrs = {'routes': routes + self._router.routes}
    self.network_client.update_router.assert_called_once_with(self._router, **attrs)
    self.assertIsNone(result)