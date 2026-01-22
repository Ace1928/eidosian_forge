from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import router
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetRouter(TestRouter):
    _default_route = {'destination': '10.20.20.0/24', 'nexthop': '10.20.30.1'}
    _network = network_fakes.create_one_network()
    _subnet = network_fakes.FakeSubnet.create_one_subnet()
    _router = network_fakes.FakeRouter.create_one_router(attrs={'routes': [_default_route], 'tags': ['green', 'red']})

    def setUp(self):
        super(TestSetRouter, self).setUp()
        self.network_client.update_router = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_router = mock.Mock(return_value=self._router)
        self.network_client.find_network = mock.Mock(return_value=self._network)
        self.network_client.find_subnet = mock.Mock(return_value=self._subnet)
        self.cmd = router.SetRouter(self.app, self.namespace)

    def test_set_this(self):
        arglist = [self._router.name, '--enable', '--distributed', '--name', 'noob', '--no-ha', '--description', 'router']
        verifylist = [('router', self._router.name), ('enable', True), ('distributed', True), ('name', 'noob'), ('description', 'router'), ('no_ha', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': True, 'distributed': True, 'name': 'noob', 'ha': False, 'description': 'router'}
        self.network_client.update_router.assert_called_once_with(self._router, **attrs)
        self.assertIsNone(result)

    def test_set_that(self):
        arglist = [self._router.name, '--disable', '--centralized', '--ha']
        verifylist = [('router', self._router.name), ('disable', True), ('centralized', True), ('ha', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'admin_state_up': False, 'distributed': False, 'ha': True}
        self.network_client.update_router.assert_called_once_with(self._router, **attrs)
        self.assertIsNone(result)

    def test_set_distributed_centralized(self):
        arglist = [self._router.name, '--distributed', '--centralized']
        verifylist = [('router', self._router.name), ('distributed', True), ('distributed', False)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_route(self):
        arglist = [self._router.name, '--route', 'destination=10.20.30.0/24,gateway=10.20.30.1']
        verifylist = [('router', self._router.name), ('routes', [{'destination': '10.20.30.0/24', 'gateway': '10.20.30.1'}])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        routes = [{'destination': '10.20.30.0/24', 'nexthop': '10.20.30.1'}]
        attrs = {'routes': routes + self._router.routes}
        self.network_client.update_router.assert_called_once_with(self._router, **attrs)
        self.assertIsNone(result)

    def test_set_no_route(self):
        arglist = [self._router.name, '--no-route']
        verifylist = [('router', self._router.name), ('no_route', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'routes': []}
        self.network_client.update_router.assert_called_once_with(self._router, **attrs)
        self.assertIsNone(result)

    def test_set_route_overwrite_route(self):
        _testrouter = network_fakes.FakeRouter.create_one_router({'routes': [{'destination': '10.0.0.2', 'nexthop': '1.1.1.1'}]})
        self.network_client.find_router = mock.Mock(return_value=_testrouter)
        arglist = [_testrouter.name, '--route', 'destination=10.20.30.0/24,gateway=10.20.30.1', '--no-route']
        verifylist = [('router', _testrouter.name), ('routes', [{'destination': '10.20.30.0/24', 'gateway': '10.20.30.1'}]), ('no_route', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'routes': [{'destination': '10.20.30.0/24', 'nexthop': '10.20.30.1'}]}
        self.network_client.update_router.assert_called_once_with(_testrouter, **attrs)
        self.assertIsNone(result)

    def test_set_nothing(self):
        arglist = [self._router.name]
        verifylist = [('router', self._router.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_router.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def test_wrong_gateway_params(self):
        arglist = ['--fixed-ip', "subnet='abc'", self._router.id]
        verifylist = [('fixed_ip', [{'subnet': "'abc'"}]), ('router', self._router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_set_gateway_network_only(self):
        arglist = ['--external-gateway', self._network.id, self._router.id]
        verifylist = [('external_gateway', self._network.id), ('router', self._router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id}})
        self.assertIsNone(result)

    def test_set_gateway_options_subnet_only(self):
        arglist = ['--external-gateway', self._network.id, '--fixed-ip', "subnet='abc'", self._router.id, '--enable-snat']
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('fixed_ip', [{'subnet': "'abc'"}]), ('enable_snat', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'external_fixed_ips': [{'subnet_id': self._subnet.id}], 'enable_snat': True}})
        self.assertIsNone(result)

    def test_set_gateway_option_ipaddress_only(self):
        arglist = ['--external-gateway', self._network.id, '--fixed-ip', 'ip-address=10.0.1.1', self._router.id, '--enable-snat']
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('fixed_ip', [{'ip-address': '10.0.1.1'}]), ('enable_snat', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'external_fixed_ips': [{'ip_address': '10.0.1.1'}], 'enable_snat': True}})
        self.assertIsNone(result)

    def test_set_gateway_options_subnet_ipaddress(self):
        arglist = ['--external-gateway', self._network.id, '--fixed-ip', "subnet='abc',ip-address=10.0.1.1", self._router.id, '--enable-snat']
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('fixed_ip', [{'subnet': "'abc'", 'ip-address': '10.0.1.1'}]), ('enable_snat', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'external_fixed_ips': [{'subnet_id': self._subnet.id, 'ip_address': '10.0.1.1'}], 'enable_snat': True}})
        self.assertIsNone(result)

    def _test_set_tags(self, with_tags=True):
        if with_tags:
            arglist = ['--tag', 'red', '--tag', 'blue']
            verifylist = [('tags', ['red', 'blue'])]
            expected_args = ['red', 'blue', 'green']
        else:
            arglist = ['--no-tag']
            verifylist = [('no_tag', True)]
            expected_args = []
        arglist.append(self._router.name)
        verifylist.append(('router', self._router.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_router.called)
        self.network_client.set_tags.assert_called_once_with(self._router, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)

    def test_set_gateway_ip_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--external-gateway', self._network.id, '--qos-policy', qos_policy.id, self._router.id]
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('qos_policy', qos_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'qos_policy_id': qos_policy.id}})
        self.assertIsNone(result)

    def test_unset_gateway_ip_qos(self):
        arglist = ['--external-gateway', self._network.id, '--no-qos-policy', self._router.id]
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('no_qos_policy', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_router.assert_called_with(self._router, **{'external_gateway_info': {'network_id': self._network.id, 'qos_policy_id': None}})
        self.assertIsNone(result)

    def test_set_unset_gateway_ip_qos(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        arglist = ['--external-gateway', self._network.id, '--qos-policy', qos_policy.id, '--no-qos-policy', self._router.id]
        verifylist = [('router', self._router.id), ('external_gateway', self._network.id), ('qos_policy', qos_policy.id), ('no_qos_policy', True)]
        self.assertRaises(tests_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_set_gateway_ip_qos_no_gateway(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        router = network_fakes.FakeRouter.create_one_router()
        self.network_client.find_router = mock.Mock(return_value=router)
        arglist = ['--qos-policy', qos_policy.id, router.id]
        verifylist = [('router', router.id), ('qos_policy', qos_policy.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)

    def test_unset_gateway_ip_qos_no_gateway(self):
        qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
        self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
        router = network_fakes.FakeRouter.create_one_router()
        self.network_client.find_router = mock.Mock(return_value=router)
        arglist = ['--no-qos-policy', router.id]
        verifylist = [('router', router.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)