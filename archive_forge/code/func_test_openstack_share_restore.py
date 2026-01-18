from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_restore(self):
    share = self.create_share(name='test_share')
    result1 = self.dict_result('share', f'show {share['id']}')
    self.assertEqual(share['id'], result1['id'])
    self.assertEqual(share['name'], result1['name'])
    self.openstack(f'share delete {share['id']} --soft')
    self.check_object_deleted('share', share['id'])
    shares_list_after_delete = self.listing_result('share', 'list')
    self.assertNotIn(share['id'], [item['ID'] for item in shares_list_after_delete])
    self.openstack(f'share restore {share['id']} ')
    shares_list_after_restore = self.listing_result('share', 'list')
    self.assertIn(share['id'], [item['ID'] for item in shares_list_after_restore])