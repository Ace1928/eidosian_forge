import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_export_location_show(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share=share['id'])
    export_location_list = self.listing_result('share snapshot export location', f'list {snapshot['id']}')
    export_location = self.dict_result('share snapshot export location', f'show {snapshot['id']} {export_location_list[0]['ID']}')
    self.assertIn('id', export_location)
    self.assertIn('created_at', export_location)
    self.assertIn('is_admin_only', export_location)
    self.assertIn('path', export_location)
    self.assertIn('share_snapshot_instance_id', export_location)
    self.assertIn('updated_at', export_location)