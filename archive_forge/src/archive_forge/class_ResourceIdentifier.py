import collections
import re
from oslo_utils import encodeutils
from urllib import parse as urlparse
from heat.common.i18n import _
class ResourceIdentifier(HeatIdentifier):
    """An identifier for a resource."""
    RESOURCE_NAME = 'resource_name'

    def __init__(self, tenant, stack_name, stack_id, path, resource_name=None):
        """Initialise a new Resource identifier.

        The identifier is based on the identifier components of
        the owning stack and the resource name.
        """
        if resource_name is not None:
            if '/' in resource_name:
                raise ValueError(_('Resource name may not contain "/"'))
            path = '/'.join([path.rstrip('/'), 'resources', resource_name])
        super(ResourceIdentifier, self).__init__(tenant, stack_name, stack_id, path)

    def __getattr__(self, attr):
        """Return a component of the identity when accessed as an attribute."""
        if attr == self.RESOURCE_NAME:
            return self._path_components()[-1]
        return HeatIdentifier.__getattr__(self, attr)

    def stack(self):
        """Return a HeatIdentifier for the owning stack."""
        return HeatIdentifier(self.tenant, self.stack_name, self.stack_id, '/'.join(self._path_components()[:-2]))