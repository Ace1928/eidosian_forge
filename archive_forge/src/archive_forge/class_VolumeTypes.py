from troveclient import base
class VolumeTypes(base.ManagerWithFind):
    """Manage :class:`VolumeType` resources."""
    resource_class = VolumeType

    def list(self):
        """Get a list of all volume-types.
        :rtype: list of :class:`VolumeType`.
        """
        return self._list('/volume-types', 'volume_types')

    def list_datastore_version_associated_volume_types(self, datastore, version_id):
        """Get a list of all volume-types for the specified datastore type
        and datastore version .
        :rtype: list of :class:`VolumeType`.
        """
        return self._list('/datastores/%s/versions/%s/volume-types' % (datastore, version_id), 'volume_types')

    def get(self, volume_type):
        """Get a specific volume-type.

        :rtype: :class:`VolumeType`
        """
        return self._get('/volume-types/%s' % base.getid(volume_type), 'volume_type')