from openstack import exceptions
from openstack.tests.functional import base
def test_set_unset_flavor_specs(self):
    """
        Test setting and unsetting flavor extra specs
        """
    if not self.operator_cloud:
        self.skipTest('Operator cloud is required for this test')
    flavor_name = self.new_item_name + '_spec_test'
    kwargs = dict(name=flavor_name, ram=1024, vcpus=2, disk=10)
    new_flavor = self.operator_cloud.create_flavor(**kwargs)
    self.assertEqual({}, new_flavor['extra_specs'])
    extra_specs = {'foo': 'aaa', 'bar': 'bbb'}
    self.operator_cloud.set_flavor_specs(new_flavor['id'], extra_specs)
    mod_flavor = self.operator_cloud.get_flavor(new_flavor['id'], get_extra=True)
    self.assertIn('extra_specs', mod_flavor)
    self.assertEqual(extra_specs, mod_flavor['extra_specs'])
    self.operator_cloud.unset_flavor_specs(mod_flavor['id'], ['foo'])
    mod_flavor = self.operator_cloud.get_flavor_by_id(new_flavor['id'], get_extra=True)
    self.assertEqual({'bar': 'bbb'}, mod_flavor['extra_specs'])