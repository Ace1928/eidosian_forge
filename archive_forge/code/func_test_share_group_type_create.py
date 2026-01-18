from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from manilaclient.tests.functional.osc import base
def test_share_group_type_create(self):
    share_group_type_name = 'test_share_group_type_create'
    share_group_type = self.create_share_group_type(name=share_group_type_name, share_types='dhss_false')
    self.assertEqual(share_group_type['name'], share_group_type_name)
    self.assertIsNotNone(share_group_type['share_types'])
    shares_group_type_list = self.listing_result('share', 'group type list')
    self.assertIn(share_group_type['id'], [item['ID'] for item in shares_group_type_list])