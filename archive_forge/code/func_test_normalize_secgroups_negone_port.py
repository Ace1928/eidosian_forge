import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_normalize_secgroups_negone_port(self):
    nova_secgroup = dict(id='abc123', name='nova_secgroup', description='A Nova security group with -1 ports', rules=[dict(id='123', from_port=-1, to_port=-1, ip_protocol='icmp', ip_range={'cidr': '0.0.0.0/0'}, parent_group_id='xyz123')])
    retval = self.cloud._normalize_secgroup(nova_secgroup)
    self.assertIsNone(retval['security_group_rules'][0]['port_range_min'])
    self.assertIsNone(retval['security_group_rules'][0]['port_range_max'])