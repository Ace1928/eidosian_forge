import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_abandon_adopt(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share=share['id'], add_cleanup=False)
    self.openstack(f'share snapshot abandon {snapshot['id']} --wait')
    snapshots_list = self.listing_result('share snapshot', 'list')
    self.assertNotIn(snapshot['id'], [item['ID'] for item in snapshots_list])
    snapshot = self.dict_result('share snapshot', f'adopt {share['id']} 10.0.0.1:/foo/path --name Snap --description Zorilla --wait')
    snapshots_list = self.listing_result('share snapshot', 'list')
    self.assertIn(snapshot['id'], [item['ID'] for item in snapshots_list])
    self.addCleanup(self.openstack, f'share snapshot delete {snapshot['id']} --force --wait')