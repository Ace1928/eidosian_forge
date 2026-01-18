from cinderclient.tests.functional import base
def test_volume_create_delete_name(self):
    """Create and delete a volume by name."""
    volume = self.object_create('volume', params='1 --name TestVolumeNamedCreate')
    self.cinder('delete', params='TestVolumeNamedCreate')
    self.check_object_deleted('volume', volume['id'])