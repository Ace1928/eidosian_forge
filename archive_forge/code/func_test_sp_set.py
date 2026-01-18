from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional.identity.v3 import common
def test_sp_set(self):
    service_provider = self._create_dummy_sp(add_clean_up=True)
    new_description = data_utils.rand_name('newDescription')
    raw_output = self.openstack('service provider set %(service-provider)s --description %(description)s ' % {'service-provider': service_provider, 'description': new_description})
    self.assertEqual(0, len(raw_output))
    raw_output = self.openstack('service provider show %s' % service_provider)
    updated_value = self.parse_show_as_object(raw_output)
    self.assertIn(new_description, updated_value['description'])