from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestShowRouter(TestRouter):
    _router = network_fakes.FakeRouter.create_one_router()
    _port = network_fakes.create_one_port({'device_owner': 'network:router_interface', 'device_id': _router.id})
    setattr(_router, 'interfaces_info', [{'port_id': _port.id, 'ip_address': _port.fixed_ips[0]['ip_address'], 'subnet_id': _port.fixed_ips[0]['subnet_id']}])
    columns = ('admin_state_up', 'availability_zone_hints', 'availability_zones', 'description', 'distributed', 'external_gateway_info', 'ha', 'id', 'interfaces_info', 'name', 'project_id', 'routes', 'status', 'tags')
    data = (router.AdminStateColumn(_router.admin_state_up), format_columns.ListColumn(_router.availability_zone_hints), format_columns.ListColumn(_router.availability_zones), _router.description, _router.distributed, router.RouterInfoColumn(_router.external_gateway_info), _router.ha, _router.id, router.RouterInfoColumn(_router.interfaces_info), _router.name, _router.project_id, router.RoutesColumn(_router.routes), _router.status, format_columns.ListColumn(_router.tags))

    def setUp(self):
        super(TestShowRouter, self).setUp()
        self.network_client.find_router = mock.Mock(return_value=self._router)
        self.network_client.ports = mock.Mock(return_value=[self._port])
        self.cmd = router.ShowRouter(self.app, self.namespace)

    def test_show_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_show_all_options(self):
        arglist = [self._router.name]
        verifylist = [('router', self._router.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.network_client.find_router.assert_called_once_with(self._router.name, ignore_missing=False)
        self.network_client.ports.assert_called_with(**{'device_id': self._router.id})
        self.assertEqual(self.columns, columns)
        self.assertCountEqual(self.data, data)

    def test_show_no_ha_no_distributed(self):
        _router = network_fakes.FakeRouter.create_one_router({'ha': None, 'distributed': None})
        arglist = [_router.name]
        verifylist = [('router', _router.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.network_client, 'find_router', return_value=_router):
            columns, data = self.cmd.take_action(parsed_args)
        self.assertNotIn('is_distributed', columns)
        self.assertNotIn('is_ha', columns)

    def test_show_no_extra_route_extension(self):
        _router = network_fakes.FakeRouter.create_one_router({'routes': None})
        arglist = [_router.name]
        verifylist = [('router', _router.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        with mock.patch.object(self.network_client, 'find_router', return_value=_router):
            columns, data = self.cmd.take_action(parsed_args)
        self.assertIn('routes', columns)
        self.assertIsNone(list(data)[columns.index('routes')].human_readable())