from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestAddExtraRoutesToRouter(TestRouter):
    _router = network_fakes.FakeRouter.create_one_router()

    def setUp(self):
        super(TestAddExtraRoutesToRouter, self).setUp()
        self.network_client.add_extra_routes_to_router = mock.Mock(return_value=self._router)
        self.cmd = router.AddExtraRoutesToRouter(self.app, self.namespace)
        self.network_client.find_router = mock.Mock(return_value=self._router)

    def test_add_no_extra_route(self):
        arglist = [self._router.id]
        verifylist = [('router', self._router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_extra_routes_to_router.assert_called_with(self._router, body={'router': {'routes': []}})
        self.assertEqual(2, len(result))

    def test_add_one_extra_route(self):
        arglist = [self._router.id, '--route', 'destination=dst1,gateway=gw1']
        verifylist = [('router', self._router.id), ('routes', [{'destination': 'dst1', 'gateway': 'gw1'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_extra_routes_to_router.assert_called_with(self._router, body={'router': {'routes': [{'destination': 'dst1', 'nexthop': 'gw1'}]}})
        self.assertEqual(2, len(result))

    def test_add_multiple_extra_routes(self):
        arglist = [self._router.id, '--route', 'destination=dst1,gateway=gw1', '--route', 'destination=dst2,gateway=gw2']
        verifylist = [('router', self._router.id), ('routes', [{'destination': 'dst1', 'gateway': 'gw1'}, {'destination': 'dst2', 'gateway': 'gw2'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.add_extra_routes_to_router.assert_called_with(self._router, body={'router': {'routes': [{'destination': 'dst1', 'nexthop': 'gw1'}, {'destination': 'dst2', 'nexthop': 'gw2'}]}})
        self.assertEqual(2, len(result))