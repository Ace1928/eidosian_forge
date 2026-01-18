import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_normalize_secgroup_rules(self):
    nova_rules = [dict(id='123', from_port=80, to_port=81, ip_protocol='tcp', ip_range={'cidr': '0.0.0.0/0'}, parent_group_id='xyz123')]
    expected = [dict(id='123', direction='ingress', ethertype='IPv4', port_range_min=80, port_range_max=81, protocol='tcp', remote_ip_prefix='0.0.0.0/0', security_group_id='xyz123', tenant_id='', project_id='', remote_group_id=None, properties={}, location=dict(region_name='RegionOne', zone=None, project=dict(domain_name='default', id='1c36b64c840a42cd9e9b931a369337f0', domain_id=None, name='admin'), cloud='_test_cloud_'))]
    retval = self.cloud._normalize_secgroup_rules(nova_rules)
    self.assertEqual(expected, retval)