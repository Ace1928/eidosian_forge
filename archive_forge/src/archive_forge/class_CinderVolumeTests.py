from cinderclient.tests.functional import base
class CinderVolumeTests(base.ClientTestBase):
    """Check of base cinder volume commands."""
    CREATE_VOLUME_PROPERTY = ('attachments', 'os-vol-tenant-attr:tenant_id', 'availability_zone', 'bootable', 'created_at', 'description', 'encrypted', 'id', 'metadata', 'name', 'size', 'status', 'user_id', 'volume_type')
    SHOW_VOLUME_PROPERTY = ('attachment_ids', 'attached_servers', 'availability_zone', 'bootable', 'created_at', 'description', 'encrypted', 'id', 'metadata', 'name', 'size', 'status', 'user_id', 'volume_type')

    def test_volume_create_delete_id(self):
        """Create and delete a volume by ID."""
        volume = self.object_create('volume', params='1')
        self.assert_object_details(self.CREATE_VOLUME_PROPERTY, volume.keys())
        self.object_delete('volume', volume['id'])
        self.check_object_deleted('volume', volume['id'])

    def test_volume_create_delete_name(self):
        """Create and delete a volume by name."""
        volume = self.object_create('volume', params='1 --name TestVolumeNamedCreate')
        self.cinder('delete', params='TestVolumeNamedCreate')
        self.check_object_deleted('volume', volume['id'])

    def test_volume_show(self):
        """Show volume details."""
        volume = self.object_create('volume', params='1 --name TestVolumeShow')
        output = self.cinder('show', params='TestVolumeShow')
        volume = self._get_property_from_output(output)
        self.assertEqual('TestVolumeShow', volume['name'])
        self.assert_object_details(self.SHOW_VOLUME_PROPERTY, volume.keys())
        self.object_delete('volume', volume['id'])
        self.check_object_deleted('volume', volume['id'])

    def test_volume_extend(self):
        """Extend a volume size."""
        volume = self.object_create('volume', params='1 --name TestVolumeExtend')
        self.cinder('extend', params='%s %s' % (volume['id'], 2))
        self.wait_for_object_status('volume', volume['id'], 'available')
        output = self.cinder('show', params=volume['id'])
        volume = self._get_property_from_output(output)
        self.assertEqual('2', volume['size'])
        self.object_delete('volume', volume['id'])
        self.check_object_deleted('volume', volume['id'])