from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
class TestSetSubnet(TestSubnet):
    _subnet = network_fakes.FakeSubnet.create_one_subnet({'tags': ['green', 'red']})

    def setUp(self):
        super(TestSetSubnet, self).setUp()
        self.network_client.update_subnet = mock.Mock(return_value=None)
        self.network_client.set_tags = mock.Mock(return_value=None)
        self.network_client.find_subnet = mock.Mock(return_value=self._subnet)
        self.cmd = subnet_v2.SetSubnet(self.app, self.namespace)

    def test_set_this(self):
        arglist = ['--name', 'new_subnet', '--dhcp', '--gateway', self._subnet.gateway_ip, self._subnet.name]
        verifylist = [('name', 'new_subnet'), ('dhcp', True), ('gateway', self._subnet.gateway_ip), ('subnet', self._subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enable_dhcp': True, 'gateway_ip': self._subnet.gateway_ip, 'name': 'new_subnet'}
        self.network_client.update_subnet.assert_called_with(self._subnet, **attrs)
        self.assertIsNone(result)

    def test_set_that(self):
        arglist = ['--name', 'new_subnet', '--no-dhcp', '--gateway', 'none', self._subnet.name]
        verifylist = [('name', 'new_subnet'), ('no_dhcp', True), ('gateway', 'none'), ('subnet', self._subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enable_dhcp': False, 'gateway_ip': None, 'name': 'new_subnet'}
        self.network_client.update_subnet.assert_called_with(self._subnet, **attrs)
        self.assertIsNone(result)

    def test_set_nothing(self):
        arglist = [self._subnet.name]
        verifylist = [('subnet', self._subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet.called)
        self.assertFalse(self.network_client.set_tags.called)
        self.assertIsNone(result)

    def test_append_options(self):
        _testsubnet = network_fakes.FakeSubnet.create_one_subnet({'dns_nameservers': ['10.0.0.1'], 'service_types': ['network:router_gateway']})
        self.network_client.find_subnet = mock.Mock(return_value=_testsubnet)
        arglist = ['--dns-nameserver', '10.0.0.2', '--service-type', 'network:floatingip_agent_gateway', _testsubnet.name]
        verifylist = [('dns_nameservers', ['10.0.0.2']), ('service_types', ['network:floatingip_agent_gateway'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'dns_nameservers': ['10.0.0.2', '10.0.0.1'], 'service_types': ['network:floatingip_agent_gateway', 'network:router_gateway']}
        self.network_client.update_subnet.assert_called_once_with(_testsubnet, **attrs)
        self.assertIsNone(result)

    def test_set_non_append_options(self):
        arglist = ['--description', 'new_description', '--dhcp', '--gateway', self._subnet.gateway_ip, self._subnet.name]
        verifylist = [('description', 'new_description'), ('dhcp', True), ('gateway', self._subnet.gateway_ip), ('subnet', self._subnet.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'enable_dhcp': True, 'gateway_ip': self._subnet.gateway_ip, 'description': 'new_description'}
        self.network_client.update_subnet.assert_called_with(self._subnet, **attrs)
        self.assertIsNone(result)

    def test_overwrite_options(self):
        _testsubnet = network_fakes.FakeSubnet.create_one_subnet({'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.200', 'end': '8.8.8.250'}], 'dns_nameservers': ['10.0.0.1']})
        self.network_client.find_subnet = mock.Mock(return_value=_testsubnet)
        arglist = ['--host-route', 'destination=10.30.30.30/24,gateway=10.30.30.1', '--no-host-route', '--allocation-pool', 'start=8.8.8.100,end=8.8.8.150', '--no-allocation-pool', '--dns-nameserver', '10.1.10.1', '--no-dns-nameservers', _testsubnet.name]
        verifylist = [('host_routes', [{'destination': '10.30.30.30/24', 'gateway': '10.30.30.1'}]), ('allocation_pools', [{'start': '8.8.8.100', 'end': '8.8.8.150'}]), ('dns_nameservers', ['10.1.10.1']), ('no_dns_nameservers', True), ('no_host_route', True), ('no_allocation_pool', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'host_routes': [{'destination': '10.30.30.30/24', 'nexthop': '10.30.30.1'}], 'allocation_pools': [{'start': '8.8.8.100', 'end': '8.8.8.150'}], 'dns_nameservers': ['10.1.10.1']}
        self.network_client.update_subnet.assert_called_once_with(_testsubnet, **attrs)
        self.assertIsNone(result)

    def test_clear_options(self):
        _testsubnet = network_fakes.FakeSubnet.create_one_subnet({'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.200', 'end': '8.8.8.250'}], 'dns_nameservers': ['10.0.0.1']})
        self.network_client.find_subnet = mock.Mock(return_value=_testsubnet)
        arglist = ['--no-host-route', '--no-allocation-pool', '--no-dns-nameservers', _testsubnet.name]
        verifylist = [('no_dns_nameservers', True), ('no_host_route', True), ('no_allocation_pool', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'host_routes': [], 'allocation_pools': [], 'dns_nameservers': []}
        self.network_client.update_subnet.assert_called_once_with(_testsubnet, **attrs)
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
        arglist.append(self._subnet.name)
        verifylist.append(('subnet', self._subnet.name))
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.assertFalse(self.network_client.update_subnet.called)
        self.network_client.set_tags.assert_called_once_with(self._subnet, tests_utils.CompareBySet(expected_args))
        self.assertIsNone(result)

    def test_set_with_tags(self):
        self._test_set_tags(with_tags=True)

    def test_set_with_no_tag(self):
        self._test_set_tags(with_tags=False)

    def test_set_segment(self):
        _net = network_fakes.create_one_network()
        _segment = network_fakes.create_one_network_segment(attrs={'network_id': _net.id})
        _subnet = network_fakes.FakeSubnet.create_one_subnet({'host_routes': [{'destination': '10.20.20.0/24', 'nexthop': '10.20.20.1'}], 'allocation_pools': [{'start': '8.8.8.200', 'end': '8.8.8.250'}], 'dns_nameservers': ['10.0.0.1'], 'network_id': _net.id, 'segment_id': None})
        self.network_client.find_subnet = mock.Mock(return_value=_subnet)
        self.network_client.find_segment = mock.Mock(return_value=_segment)
        arglist = ['--network-segment', _segment.id, _subnet.name]
        verifylist = [('network_segment', _segment.id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'segment_id': _segment.id}
        self.network_client.update_subnet.assert_called_once_with(_subnet, **attrs)
        self.network_client.update_subnet.assert_called_with(_subnet, **attrs)
        self.assertIsNone(result)