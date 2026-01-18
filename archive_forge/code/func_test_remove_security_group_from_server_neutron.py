import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_remove_security_group_from_server_neutron(self):
    fake_server = fakes.make_fake_server('1234', 'server-name', 'ACTIVE')
    self.cloud.secgroup_source = 'neutron'
    validate = {'removeSecurityGroup': {'name': 'neutron-sec-group'}}
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail']), json={'servers': [fake_server]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]}), dict(method='POST', uri='%s/servers/%s/action' % (fakes.COMPUTE_ENDPOINT, '1234'), validate=dict(json=validate))])
    self.assertTrue(self.cloud.remove_server_security_groups('server-name', 'neutron-sec-group'))
    self.assert_calls()