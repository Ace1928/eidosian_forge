from openstack import exceptions
from openstack.tests.functional import base
def test_range_search_multi_4(self):
    flavors = self.user_cloud.list_flavors(get_extra=False)
    result = self.user_cloud.range_search(flavors, {'ram': '>=4096', 'vcpus': 'MAX'})
    self.assertIsInstance(result, list)
    self.assertEqual(1, len(result))
    self.assertEqual('m1.xlarge', result[0]['name'])