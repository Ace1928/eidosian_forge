from oslo_serialization import jsonutils
from tempest.lib.common.utils import data_utils
from manilaclient.tests.functional.osc import base
def test_openstack_share_group_type_list(self):
    share_group_type_name = data_utils.rand_name('test_share_group_type_create')
    share_group_type = self.create_share_group_type(name=share_group_type_name, share_types='dhss_false')
    shares_group_type_list = self.listing_result('share', 'group type list')
    self.assertTableStruct(shares_group_type_list, ['ID', 'Name', 'Share Types', 'Visibility', 'Is Default', 'Group Specs'])
    self.assertIn(share_group_type['id'], [item['ID'] for item in shares_group_type_list])