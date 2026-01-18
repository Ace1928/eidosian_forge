import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_list(self):
    share = self.create_share()
    snapshot_1 = self.create_snapshot(share=share['id'])
    snapshot_2 = self.create_snapshot(share=share['id'], description='Description')
    snapshots_list = self.listing_result('share snapshot', f'list --name {snapshot_2['name']} --description {snapshot_2['description']} --all-projects')
    self.assertTableStruct(snapshots_list, ['ID', 'Name', 'Project ID'])
    self.assertIn(snapshot_2['name'], [snap['Name'] for snap in snapshots_list])
    self.assertEqual(1, len(snapshots_list))
    snapshots_list = self.listing_result('share snapshot', f'list --share {share['name']}')
    self.assertTableStruct(snapshots_list, ['ID', 'Name'])
    id_list = [snap['ID'] for snap in snapshots_list]
    self.assertIn(snapshot_1['id'], id_list)
    self.assertIn(snapshot_2['id'], id_list)
    snapshots_list = self.listing_result('share snapshot', f'list --name~ {snapshot_2['name'][-3:]} --description~ Des --detail')
    self.assertTableStruct(snapshots_list, ['ID', 'Name', 'Status', 'Description', 'Created At', 'Size', 'Share ID', 'Share Proto', 'Share Size', 'User ID'])
    self.assertIn(snapshot_2['name'], [snap['Name'] for snap in snapshots_list])
    self.assertEqual(snapshot_2['description'], snapshots_list[0]['Description'])
    self.assertEqual(1, len(snapshots_list))