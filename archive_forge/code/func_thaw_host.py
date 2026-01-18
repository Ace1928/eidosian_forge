from cinderclient import api_versions
from cinderclient import base
def thaw_host(self, host):
    """Thaw the service specified by hostname."""
    body = {'host': host}
    return self._update('/os-services/thaw', body)