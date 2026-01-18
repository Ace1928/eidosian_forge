import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_export_location_list(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share=share['id'])
    export_location_list = self.listing_result('share snapshot export location', f' list {snapshot['id']}')
    self.assertTableStruct(export_location_list, ['ID', 'Path'])