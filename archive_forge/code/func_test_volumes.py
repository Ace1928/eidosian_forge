from fixtures import TimeoutException
from testtools import content
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_volumes(self):
    """Test volume and snapshot functionality"""
    volume_name = self.getUniqueString()
    snapshot_name = self.getUniqueString()
    self.addDetail('volume', content.text_content(volume_name))
    self.addCleanup(self.cleanup, volume_name, snapshot_name=snapshot_name)
    volume = self.user_cloud.create_volume(display_name=volume_name, size=1)
    snapshot = self.user_cloud.create_volume_snapshot(volume['id'], display_name=snapshot_name)
    ret_volume = self.user_cloud.get_volume_by_id(volume['id'])
    self.assertEqual(volume['id'], ret_volume['id'])
    volume_ids = [v['id'] for v in self.user_cloud.list_volumes()]
    self.assertIn(volume['id'], volume_ids)
    snapshot_list = self.user_cloud.list_volume_snapshots()
    snapshot_ids = [s['id'] for s in snapshot_list]
    self.assertIn(snapshot['id'], snapshot_ids)
    ret_snapshot = self.user_cloud.get_volume_snapshot_by_id(snapshot['id'])
    self.assertEqual(snapshot['id'], ret_snapshot['id'])
    self.user_cloud.delete_volume_snapshot(snapshot_name, wait=True)
    self.user_cloud.delete_volume(volume_name, wait=True)