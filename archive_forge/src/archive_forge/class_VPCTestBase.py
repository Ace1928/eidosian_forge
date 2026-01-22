from unittest import mock
import uuid
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine.resources.aws.ec2 import subnet as sn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class VPCTestBase(common.HeatTestCase):

    def setUp(self):
        super(VPCTestBase, self).setUp()
        self.mockclient = mock.Mock(spec=neutronclient.Client)
        self.patchobject(neutronclient, 'Client', return_value=self.mockclient)
        self.mockclient.add_interface_router.return_value = None
        self.mockclient.delete_network.return_value = None
        self.mockclient.delete_port.return_value = None
        self.mockclient.delete_router.return_value = None
        self.mockclient.delete_subnet.return_value = None
        self.mockclient.remove_interface_router.return_value = None
        self.mockclient.remove_gateway_router.return_value = None
        self.mockclient.delete_security_group_rule.return_value = None
        self.mockclient.delete_security_group.return_value = None
        self.vpc_name = utils.PhysName('test_stack', 'the_vpc')
        self.mock_router_for_vpc()
        self.subnet_name = utils.PhysName('test_stack', 'the_subnet')
        self.mock_show_subnet()
        self.stub_SubnetConstraint_validate()
        self.mock_create_subnet()

    def create_stack(self, templ):
        t = template_format.parse(templ)
        stack = self.parse_stack(t)
        self.assertIsNone(stack.validate())
        self.assertIsNone(stack.create())
        return stack

    def parse_stack(self, t):
        stack_name = 'test_stack'
        tmpl = template.Template(t)
        stack = parser.Stack(utils.dummy_context(), stack_name, tmpl)
        stack.store()
        return stack

    def validate_mock_create_network(self):
        self.mockclient.show_network.assert_called_with('aaaa')
        self.mockclient.create_network.assert_called_once_with({'network': {'name': self.vpc_name}})
        self.mockclient.create_router.assert_called_once()

    def mock_create_network(self):
        self.mockclient.create_network.return_value = {'network': {'status': 'BUILD', 'subnets': [], 'name': 'name', 'admin_state_up': True, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}}
        show_network_returns = [{'network': {'status': 'BUILD', 'subnets': [], 'name': self.vpc_name, 'admin_state_up': False, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}}]
        for i in range(7):
            show_network_returns.append({'network': {'status': 'ACTIVE', 'subnets': [], 'name': self.vpc_name, 'admin_state_up': False, 'shared': False, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'aaaa'}})
        self.mockclient.show_network.side_effect = show_network_returns
        self.mockclient.create_router.return_value = {'router': {'status': 'BUILD', 'name': self.vpc_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'bbbb'}}

    def mock_create_subnet(self):
        self.subnet_name = utils.PhysName('test_stack', 'the_subnet')
        self.mockclient.create_subnet.return_value = {'subnet': {'status': 'ACTIVE', 'name': self.subnet_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'cccc'}}

    def mock_show_subnet(self):
        self.mockclient.show_subnet.return_value = {'subnet': {'name': self.subnet_name, 'network_id': 'aaaa', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'allocation_pools': [{'start': '10.0.0.2', 'end': '10.0.0.254'}], 'gateway_ip': '10.0.0.1', 'ip_version': 4, 'cidr': '10.0.0.0/24', 'id': 'cccc', 'enable_dhcp': False}}

    def mock_create_security_group(self):
        self.sg_name = utils.PhysName('test_stack', 'the_sg')
        self.mockclient.create_security_group.return_value = {'security_group': {'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'name': self.sg_name, 'description': 'SSH access', 'security_group_rules': [], 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}
        self.create_security_group_rule_expected = {'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}
        self.mockclient.create_security_group_rule.return_value = {'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': 22, 'ethertype': 'IPv4', 'port_range_max': 22, 'protocol': 'tcp', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'id': 'bbbb'}}

    def mock_show_security_group(self):
        sg_name = utils.PhysName('test_stack', 'the_sg')
        self._group = '0389f747-7785-4757-b7bb-2ab07e4b09c3'
        self.mockclient.show_security_group.return_value = {'security_group': {'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'name': sg_name, 'description': '', 'security_group_rules': [{'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'bbbb', 'ethertype': 'IPv4', 'security_group_id': '0389f747-7785-4757-b7bb-2ab07e4b09c3', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'port_range_min': 22}], 'id': '0389f747-7785-4757-b7bb-2ab07e4b09c3'}}

    def mock_router_for_vpc(self):
        self.mockclient.list_routers.return_value = {'routers': [{'status': 'ACTIVE', 'external_gateway_info': {'network_id': 'zzzz', 'enable_snat': True}, 'name': self.vpc_name, 'admin_state_up': True, 'tenant_id': '3e21026f2dc94372b105808c0e721661', 'routes': [], 'id': 'bbbb'}]}

    def mock_create_route_table(self):
        self.rt_name = utils.PhysName('test_stack', 'the_route_table')
        self.mockclient.create_router.return_value = {'router': {'status': 'BUILD', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}}
        show_router_returns = [{'router': {'status': 'BUILD', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}}]
        for i in range(3):
            show_router_returns.append({'router': {'status': 'ACTIVE', 'name': self.rt_name, 'admin_state_up': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'id': 'ffff'}})
        self.mockclient.show_router.side_effect = show_router_returns

    def assertResourceState(self, resource, ref_id):
        self.assertIsNone(resource.validate())
        self.assertEqual((resource.CREATE, resource.COMPLETE), resource.state)
        self.assertEqual(ref_id, resource.FnGetRefId())