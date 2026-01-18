import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_update_security_group_nova(self):
    self.has_neutron = False
    new_name = self.getUniqueString()
    self.cloud.secgroup_source = 'nova'
    nova_return = [nova_grp_dict]
    update_return = nova_grp_dict.copy()
    update_return['name'] = new_name
    self.register_uris([dict(method='GET', uri='{endpoint}/os-security-groups'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_groups': nova_return}), dict(method='PUT', uri='{endpoint}/os-security-groups/2'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'security_group': update_return})])
    r = self.cloud.update_security_group(nova_grp_dict['id'], name=new_name)
    self.assertEqual(r['name'], new_name)
    self.assert_calls()