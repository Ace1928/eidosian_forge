import collections
import copy
from unittest import mock
from neutronclient.common import exceptions as neutron_exc
from neutronclient.v2_0 import client as neutronclient
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_security_group_neutron_exception(self):
    self.m_csg.return_value = {'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': self.sg_name, 'description': 'HTTP and SSH access', 'security_group_rules': [], 'id': 'aaaa'}}
    self.m_csgr.side_effect = neutron_exc.Conflict
    self.m_dsgr.side_effect = neutron_exc.NeutronClientException(status_code=404)
    self.m_ssg.side_effect = [{'security_group': {'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'name': 'sc1', 'description': '', 'security_group_rules': [{'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'bbbb', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 22}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': 80, 'id': 'cccc', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '0.0.0.0/0', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 80}, {'direction': 'ingress', 'protocol': 'tcp', 'port_range_max': None, 'id': 'dddd', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': 'wwww', 'remote_ip_prefix': None, 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}, {'direction': 'egress', 'protocol': 'tcp', 'port_range_max': 22, 'id': 'eeee', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': '10.0.1.0/24', 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': 22}, {'direction': 'egress', 'protocol': None, 'port_range_max': None, 'id': 'ffff', 'ethertype': 'IPv4', 'security_group_id': 'aaaa', 'remote_group_id': None, 'remote_ip_prefix': None, 'tenant_id': 'f18ca530cc05425e8bac0a5ff92f7e88', 'port_range_min': None}], 'id': 'aaaa'}}, neutron_exc.NeutronClientException(status_code=404)]
    stack = self.create_stack(self.test_template_neutron)
    sg = stack['the_sg']
    self.assertResourceState(sg, 'aaaa')
    scheduler.TaskRunner(sg.delete)()
    sg.state_set(sg.CREATE, sg.COMPLETE, 'to delete again')
    sg.resource_id = 'aaaa'
    stack.delete()
    self.m_csg.assert_called_once_with({'security_group': {'name': self.sg_name, 'description': 'HTTP and SSH access'}})
    self.validate_create_security_group_rule_calls()
    self.assertEqual([mock.call('aaaa'), mock.call('aaaa')], self.m_ssg.call_args_list)
    self.assertEqual([mock.call('bbbb'), mock.call('cccc'), mock.call('dddd'), mock.call('eeee'), mock.call('ffff')], self.m_dsgr.call_args_list)