from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_security_group_exception(self):
    sg_name = utils.PhysName('test_stack', 'the_sg')
    self.mockclient.create_security_group.return_value = {'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': sg_name, 'description': 'HTTP and SSH access', 'security_group_rules': [], 'id': 'aaaa'}}
    self.mockclient.create_security_group_rule.side_effect = [neutron_exc.Conflict, neutron_exc.Conflict, neutron_exc.Conflict, neutron_exc.Conflict, neutron_exc.Conflict, neutron_exc.Conflict]
    self.mockclient.show_security_group.side_effect = [{'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': sg_name, 'description': 'HTTP and SSH access', 'security_group_rules': [], 'id': 'aaaa'}}, {'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': 'sc1', 'description': '', 'security_group_rules': [{'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': '22', 'id': 'bbbb', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': '22'}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': '80', 'id': 'cccc', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': '80'}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': None, 'id': 'dddd', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': 'wwww', 'remote_ip_prefix': None, 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}, {'direction': 'egress', 'protocol': 'tcp', 'port_range_max': '22', 'id': 'eeee', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '10.0.1.0/24', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': '22'}, {'direction': 'egress', 'protocol': None, 'port_range_max': None, 'id': 'ffff', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': 'xxxx', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}, {'direction': 'egress', 'protocol': None, 'port_range_max': None, 'id': 'gggg', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': 'aaaa', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}], 'id': 'aaaa'}}, neutron_exc.NeutronClientException(status_code=404)]
    self.mockclient.delete_security_group_rule.side_effect = neutron_exc.NeutronClientException(status_code=404)
    self.mockclient.delete_security_group.side_effect = neutron_exc.NeutronClientException(status_code=404)
    stack = self.create_stack(self.test_template)
    sg = stack['the_sg']
    self.assertResourceState(sg, 'aaaa')
    scheduler.TaskRunner(sg.delete)()
    sg.state_set(sg.CREATE, sg.COMPLETE, 'to delete again')
    sg.resource_id = 'aaaa'
    stack.delete()
    self.mockclient.create_security_group.assert_called_once_with({'security_group': {'name': sg_name, 'description': 'HTTP and SSH access'}})
    self.mockclient.create_security_group_rule.assert_has_calls([mock.call({'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': '22', 'ethertype': 'IPv4', 'port_range_max': '22', 'protocol': 'tcp', 'security_group_id': 'aaaa'}}), mock.call({'security_group_rule': {'direction': 'ingress', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'port_range_min': '80', 'ethertype': 'IPv4', 'port_range_max': '80', 'protocol': 'tcp', 'security_group_id': 'aaaa'}}), mock.call({'security_group_rule': {'direction': 'ingress', 'remote_group_id': 'wwww', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': 'tcp', 'security_group_id': 'aaaa'}}), mock.call({'security_group_rule': {'direction': 'egress', 'remote_group_id': None, 'remote_ip_prefix': '10.0.1.0/24', 'port_range_min': '22', 'ethertype': 'IPv4', 'port_range_max': '22', 'protocol': 'tcp', 'security_group_id': 'aaaa'}}), mock.call({'security_group_rule': {'direction': 'egress', 'remote_group_id': 'xxxx', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': None, 'security_group_id': 'aaaa'}}), mock.call({'security_group_rule': {'direction': 'egress', 'remote_group_id': 'aaaa', 'remote_ip_prefix': None, 'port_range_min': None, 'ethertype': 'IPv4', 'port_range_max': None, 'protocol': None, 'security_group_id': 'aaaa'}})])
    self.mockclient.show_security_group.assert_called_with('aaaa')
    self.mockclient.delete_security_group_rule.assert_has_calls([mock.call('bbbb'), mock.call('cccc'), mock.call('dddd'), mock.call('eeee'), mock.call('ffff'), mock.call('gggg')])
    self.mockclient.delete_security_group.assert_called_with('aaaa')