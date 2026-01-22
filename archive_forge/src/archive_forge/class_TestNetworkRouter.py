from unittest import mock
import uuid
from openstack import exceptions
from openstack.network.v2 import _proxy
from openstack.network.v2 import address_group
from openstack.network.v2 import address_scope
from openstack.network.v2 import agent
from openstack.network.v2 import auto_allocated_topology
from openstack.network.v2 import availability_zone
from openstack.network.v2 import bgp_peer
from openstack.network.v2 import bgp_speaker
from openstack.network.v2 import bgpvpn
from openstack.network.v2 import bgpvpn_network_association
from openstack.network.v2 import bgpvpn_port_association
from openstack.network.v2 import bgpvpn_router_association
from openstack.network.v2 import extension
from openstack.network.v2 import firewall_group
from openstack.network.v2 import firewall_policy
from openstack.network.v2 import firewall_rule
from openstack.network.v2 import flavor
from openstack.network.v2 import floating_ip
from openstack.network.v2 import health_monitor
from openstack.network.v2 import l3_conntrack_helper
from openstack.network.v2 import listener
from openstack.network.v2 import load_balancer
from openstack.network.v2 import local_ip
from openstack.network.v2 import local_ip_association
from openstack.network.v2 import metering_label
from openstack.network.v2 import metering_label_rule
from openstack.network.v2 import ndp_proxy
from openstack.network.v2 import network
from openstack.network.v2 import network_ip_availability
from openstack.network.v2 import network_segment_range
from openstack.network.v2 import pool
from openstack.network.v2 import pool_member
from openstack.network.v2 import port
from openstack.network.v2 import port_forwarding
from openstack.network.v2 import qos_bandwidth_limit_rule
from openstack.network.v2 import qos_dscp_marking_rule
from openstack.network.v2 import qos_minimum_bandwidth_rule
from openstack.network.v2 import qos_minimum_packet_rate_rule
from openstack.network.v2 import qos_policy
from openstack.network.v2 import qos_rule_type
from openstack.network.v2 import quota
from openstack.network.v2 import rbac_policy
from openstack.network.v2 import router
from openstack.network.v2 import security_group
from openstack.network.v2 import security_group_rule
from openstack.network.v2 import segment
from openstack.network.v2 import service_profile
from openstack.network.v2 import service_provider
from openstack.network.v2 import subnet
from openstack.network.v2 import subnet_pool
from openstack.network.v2 import vpn_endpoint_group
from openstack.network.v2 import vpn_ike_policy
from openstack.network.v2 import vpn_ipsec_policy
from openstack.network.v2 import vpn_ipsec_site_connection
from openstack.network.v2 import vpn_service
from openstack import proxy as proxy_base
from openstack.tests.unit import test_proxy_base
class TestNetworkRouter(TestNetworkProxy):

    def test_router_create_attrs(self):
        self.verify_create(self.proxy.create_router, router.Router)

    def test_router_delete(self):
        self.verify_delete(self.proxy.delete_router, router.Router, False, expected_kwargs={'if_revision': None})

    def test_router_delete_ignore(self):
        self.verify_delete(self.proxy.delete_router, router.Router, True, expected_kwargs={'if_revision': None})

    def test_router_delete_if_revision(self):
        self.verify_delete(self.proxy.delete_router, router.Router, True, method_kwargs={'if_revision': 42}, expected_kwargs={'if_revision': 42})

    def test_router_find(self):
        self.verify_find(self.proxy.find_router, router.Router)

    def test_router_get(self):
        self.verify_get(self.proxy.get_router, router.Router)

    def test_routers(self):
        self.verify_list(self.proxy.routers, router.Router)

    def test_router_update(self):
        self.verify_update(self.proxy.update_router, router.Router, expected_kwargs={'x': 1, 'y': 2, 'z': 3, 'if_revision': None})

    def test_router_update_if_revision(self):
        self.verify_update(self.proxy.update_router, router.Router, method_kwargs={'x': 1, 'y': 2, 'z': 3, 'if_revision': 42}, expected_kwargs={'x': 1, 'y': 2, 'z': 3, 'if_revision': 42})

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'add_interface')
    def test_add_interface_to_router_with_port(self, mock_add_interface, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.add_interface', self.proxy.add_interface_to_router, method_args=['FAKE_ROUTER'], method_kwargs={'port_id': 'PORT'}, expected_args=[self.proxy], expected_kwargs={'port_id': 'PORT'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'add_interface')
    def test_add_interface_to_router_with_subnet(self, mock_add_interface, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.add_interface', self.proxy.add_interface_to_router, method_args=['FAKE_ROUTER'], method_kwargs={'subnet_id': 'SUBNET'}, expected_args=[self.proxy], expected_kwargs={'subnet_id': 'SUBNET'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'remove_interface')
    def test_remove_interface_from_router_with_port(self, mock_remove, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.remove_interface', self.proxy.remove_interface_from_router, method_args=['FAKE_ROUTER'], method_kwargs={'port_id': 'PORT'}, expected_args=[self.proxy], expected_kwargs={'port_id': 'PORT'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'remove_interface')
    def test_remove_interface_from_router_with_subnet(self, mock_remove, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.remove_interface', self.proxy.remove_interface_from_router, method_args=['FAKE_ROUTER'], method_kwargs={'subnet_id': 'SUBNET'}, expected_args=[self.proxy], expected_kwargs={'subnet_id': 'SUBNET'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'add_extra_routes')
    def test_add_extra_routes_to_router(self, mock_add_extra_routes, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.add_extra_routes', self.proxy.add_extra_routes_to_router, method_args=['FAKE_ROUTER'], method_kwargs={'body': {'router': {'routes': []}}}, expected_args=[self.proxy], expected_kwargs={'body': {'router': {'routes': []}}})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'remove_extra_routes')
    def test_remove_extra_routes_from_router(self, mock_remove_extra_routes, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.remove_extra_routes', self.proxy.remove_extra_routes_from_router, method_args=['FAKE_ROUTER'], method_kwargs={'body': {'router': {'routes': []}}}, expected_args=[self.proxy], expected_kwargs={'body': {'router': {'routes': []}}})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'add_gateway')
    def test_add_gateway_to_router(self, mock_add, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.add_gateway', self.proxy.add_gateway_to_router, method_args=['FAKE_ROUTER'], method_kwargs={'foo': 'bar'}, expected_args=[self.proxy], expected_kwargs={'foo': 'bar'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'remove_gateway')
    def test_remove_gateway_from_router(self, mock_remove, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.remove_gateway', self.proxy.remove_gateway_from_router, method_args=['FAKE_ROUTER'], method_kwargs={'foo': 'bar'}, expected_args=[self.proxy], expected_kwargs={'foo': 'bar'})
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'add_external_gateways')
    def test_add_external_gateways(self, mock_add, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.add_external_gateways', self.proxy.add_external_gateways, method_args=['FAKE_ROUTER', 'bar'], expected_args=[self.proxy, 'bar'])
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'update_external_gateways')
    def test_update_external_gateways(self, mock_remove, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.update_external_gateways', self.proxy.update_external_gateways, method_args=['FAKE_ROUTER', 'bar'], expected_args=[self.proxy, 'bar'])
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    @mock.patch.object(proxy_base.Proxy, '_get_resource')
    @mock.patch.object(router.Router, 'remove_external_gateways')
    def test_remove_external_gateways(self, mock_remove, mock_get):
        x_router = router.Router.new(id='ROUTER_ID')
        mock_get.return_value = x_router
        self._verify('openstack.network.v2.router.Router.remove_external_gateways', self.proxy.remove_external_gateways, method_args=['FAKE_ROUTER', 'bar'], expected_args=[self.proxy, 'bar'])
        mock_get.assert_called_once_with(router.Router, 'FAKE_ROUTER')

    def test_router_hosting_l3_agents_list(self):
        self.verify_list(self.proxy.routers_hosting_l3_agents, agent.RouterL3Agent, method_kwargs={'router': ROUTER_ID}, expected_kwargs={'router_id': ROUTER_ID})

    def test_agent_hosted_routers_list(self):
        self.verify_list(self.proxy.agent_hosted_routers, router.L3AgentRouter, method_kwargs={'agent': AGENT_ID}, expected_kwargs={'agent_id': AGENT_ID})