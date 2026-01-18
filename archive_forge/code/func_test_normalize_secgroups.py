import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_normalize_secgroups(self):
    nova_secgroup = dict(id='abc123', name='nova_secgroup', description='A Nova security group', rules=[dict(id='123', from_port=80, to_port=81, ip_protocol='tcp', ip_range={'cidr': '0.0.0.0/0'}, parent_group_id='xyz123')])
    expected = dict(id='abc123', name='nova_secgroup', description='A Nova security group', project_id='', tenant_id='', properties={}, location=dict(region_name='RegionOne', zone=None, project=dict(domain_name='default', id='1c36b64c840a42cd9e9b931a369337f0', domain_id=None, name='admin'), cloud='_test_cloud_'), security_group_rules=[dict(id='123', direction='ingress', ethertype='IPv4', port_range_min=80, port_range_max=81, protocol='tcp', remote_ip_prefix='0.0.0.0/0', security_group_id='xyz123', project_id='', tenant_id='', properties={}, remote_group_id=None, location=dict(region_name='RegionOne', zone=None, project=dict(domain_name='default', id='1c36b64c840a42cd9e9b931a369337f0', domain_id=None, name='admin'), cloud='_test_cloud_'))])
    self.cloud.secgroup_source = 'nova'
    retval = self.cloud._normalize_secgroup(nova_secgroup)
    self.cloud.secgroup_source = 'neutron'
    self.assertEqual(expected, retval)