import json
from manilaclient.tests.functional.osc import base
def test_share_snapshot_unset(self):
    share = self.create_share()
    snapshot = self.create_snapshot(share=share['id'], name='Snap', description='Description')
    self.openstack(f'share snapshot unset {snapshot['id']} --name --description')
    show_result = json.loads(self.openstack(f'share snapshot show -f json {snapshot['id']}'))
    self.assertEqual(snapshot['id'], show_result['id'])
    self.assertIsNone(show_result['name'])
    self.assertIsNone(show_result['description'])