from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_export_location_show(self):
    share = self.create_share()
    share_export_locations = self.get_share_export_locations(share['id'])
    result_export_locations = self.listing_result('share', f'export location list {share['id']}')
    for share_export in share_export_locations:
        export_location = self.dict_result('share', f'export location show {share['id']} {share_export['ID']}')
        self.assertIn(export_location['id'], [item['ID'] for item in result_export_locations])