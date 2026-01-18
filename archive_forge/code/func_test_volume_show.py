from cinderclient.tests.functional import base
def test_volume_show(self):
    """Show volume details."""
    volume = self.object_create('volume', params='1 --name TestVolumeShow')
    output = self.cinder('show', params='TestVolumeShow')
    volume = self._get_property_from_output(output)
    self.assertEqual('TestVolumeShow', volume['name'])
    self.assert_object_details(self.SHOW_VOLUME_PROPERTY, volume.keys())
    self.object_delete('volume', volume['id'])
    self.check_object_deleted('volume', volume['id'])