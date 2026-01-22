from cinderclient import api_versions
from cinderclient import base
class ResourceFilterManager(base.ManagerWithFind):
    """Manage :class:`ResourceFilter` resources."""
    resource_class = ResourceFilter

    @api_versions.wraps('3.33')
    def list(self, resource=None):
        """List all resource filters."""
        url = '/resource_filters'
        if resource is not None:
            url += '?resource=%s' % resource
        return self._list(url, 'resource_filters')