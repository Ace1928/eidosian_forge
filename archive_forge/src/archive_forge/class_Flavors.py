from troveclient import base
class Flavors(base.ManagerWithFind):
    """Manage :class:`Flavor` resources."""
    resource_class = Flavor

    def list(self):
        """Get a list of all flavors.
        :rtype: list of :class:`Flavor`.
        """
        return self._list('/flavors', 'flavors')

    def list_datastore_version_associated_flavors(self, datastore, version_id):
        """Get a list of all flavors for the specified datastore type
        and datastore version .
        :rtype: list of :class:`Flavor`.
        """
        return self._list('/datastores/%s/versions/%s/flavors' % (datastore, version_id), 'flavors')

    def get(self, flavor):
        """Get a specific flavor.

        :rtype: :class:`Flavor`
        """
        return self._get('/flavors/%s' % base.getid(flavor), 'flavor')