from troveclient import base
from troveclient import common
from troveclient.v1 import users
def _is_root_enabled(self, uri):
    """Return whether root is enabled for the instance or the cluster."""
    resp, body = self.api.client.get(uri)
    common.check_for_exceptions(resp, body, uri)
    return self.resource_class(self, body, loaded=True)