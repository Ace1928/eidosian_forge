from manilaclient import api_versions
from manilaclient import base
def _do_disable(self, host, binary, resource_path=RESOURCE_PATH, disable_reason=None):
    """Disable the service specified by hostname and binary."""
    body = {'host': host, 'binary': binary}
    if disable_reason:
        body['disabled_reason'] = disable_reason
    return self._update('%s/disable' % resource_path, body)