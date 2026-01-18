from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_show(self):
    share = self.create_share()
    result = self.dict_result('share', 'show %s' % share['id'])
    self.assertEqual(share['id'], result['id'])
    listing_result = self.listing_result('share', 'show %s' % share['id'])
    self.assertTableStruct(listing_result, ['Field', 'Value'])