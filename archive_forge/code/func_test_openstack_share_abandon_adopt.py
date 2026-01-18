from manilaclient.tests.functional.osc import base
from tempest.lib.common.utils import data_utils
def test_openstack_share_abandon_adopt(self):
    share = self.create_share(add_cleanup=False)
    shares_list = self.listing_result('share', 'list')
    self.assertIn(share['id'], [item['ID'] for item in shares_list])
    export_location_obj = self.get_share_export_locations(share['id'])[0]
    export_location = export_location_obj['Path']
    source = self.dict_result('share', f'show {share['id']}')
    host = source['host']
    protocol = source['share_proto']
    share_type = source['share_type']
    self.openstack(f'share abandon {share['id']} --wait')
    self.check_object_deleted('share', share['id'])
    shares_list_after_delete = self.listing_result('share', 'list')
    self.assertNotIn(share['id'], [item['ID'] for item in shares_list_after_delete])
    result = self.dict_result('share', f'adopt {host} {protocol} {export_location} --share-type {share_type} --wait')
    self.assertEqual(host, result['host'])
    self.assertEqual(protocol, result['share_proto'])
    self.openstack(f'share delete {result['id']} --wait')