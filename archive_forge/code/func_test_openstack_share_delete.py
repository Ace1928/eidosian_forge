from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_delete(self):
    share = self.create_share(add_cleanup=False)
    shares_list = self.listing_result('share', 'list')
    self.assertIn(share['id'], [item['ID'] for item in shares_list])
    self.openstack('share delete %s' % share['id'])
    self.check_object_deleted('share', share['id'])
    shares_list_after_delete = self.listing_result('share', 'list')
    self.assertNotIn(share['id'], [item['ID'] for item in shares_list_after_delete])