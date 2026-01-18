from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_list(self):
    share = self.create_share()
    shares_list = self.listing_result('share', 'list')
    self.assertTableStruct(shares_list, ['ID', 'Name', 'Size', 'Share Proto', 'Status', 'Is Public', 'Share Type Name', 'Host', 'Availability Zone'])
    self.assertIn(share['id'], [item['ID'] for item in shares_list])