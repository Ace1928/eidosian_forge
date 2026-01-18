from openstack.block_storage.v3 import volume as _volume
from openstack.tests.functional.block_storage.v3 import base
def test_volume(self):
    volume = self.user_cloud.block_storage.get_volume(self.volume.id)
    self.assertEqual(self.volume.name, volume.name)
    volume = self.user_cloud.block_storage.find_volume(self.volume.name)
    self.assertEqual(self.volume.id, volume.id)
    volumes = self.user_cloud.block_storage.volumes()
    self.assertIn(self.volume.id, {v.id for v in volumes})
    volume_name = self.getUniqueString()
    volume_description = self.getUniqueString()
    volume = self.user_cloud.block_storage.update_volume(self.volume, name=volume_name, description=volume_description)
    self.assertIsInstance(volume, _volume.Volume)
    volume = self.user_cloud.block_storage.get_volume(self.volume.id)
    self.assertEqual(volume_name, volume.name)
    self.assertEqual(volume_description, volume.description)