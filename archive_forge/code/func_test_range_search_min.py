from openstack import exceptions
from openstack.tests.functional import base
def test_range_search_min(self):
    flavors = self.user_cloud.list_flavors(get_extra=False)
    result = self.user_cloud.range_search(flavors, {'ram': 'MIN'})
    self.assertIsInstance(result, list)
    self.assertEqual(1, len(result))
    self.assertIn(result[0]['name'], ('cirros256', 'm1.tiny'))