from manilaclient.tests.functional.osc import base
def test_openstack_share_snapshot_instance_show(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share['id'], wait=True)
    share_snapshot_instance = self.listing_result('share snapshot instance', f'list --snapshot {snapshot['id']}')
    result = self.dict_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
    self.assertEqual(share_snapshot_instance[0]['ID'], result['id'])
    self.assertEqual(share_snapshot_instance[0]['Snapshot ID'], result['snapshot_id'])
    listing_result = self.listing_result('share snapshot instance', f'show {share_snapshot_instance[0]['ID']}')
    self.assertTableStruct(listing_result, ['Field', 'Value'])