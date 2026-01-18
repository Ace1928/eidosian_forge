import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_security_group_neutron(self):
    self.cloud.secgroup_source = 'neutron'
    new_name = self.getUniqueString()
    sg_id = neutron_grp_dict['id']
    update_return = neutron_grp_dict.copy()
    update_return['name'] = new_name
    update_return['stateful'] = False
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups']), json={'security_groups': [neutron_grp_dict]}), dict(method='PUT', uri=self.get_mock_url('network', 'public', append=['v2.0', 'security-groups', '%s' % sg_id]), json={'security_group': update_return}, validate=dict(json={'security_group': {'name': new_name, 'stateful': False}}))])
    r = self.cloud.update_security_group(sg_id, name=new_name, stateful=False)
    self.assertEqual(r['name'], new_name)
    self.assertEqual(r['stateful'], False)
    self.assert_calls()