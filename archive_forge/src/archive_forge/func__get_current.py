from urllib import parse
from novaclient import base
from novaclient import exceptions as exc
def _get_current(self):
    """Returns info about current version."""
    try:
        url = '%s' % self.api.client.get_endpoint()
        return self._get(url, 'version')
    except exc.NotFound:
        url = '%s/' % url.rsplit('/', 1)[0]
        return self._get(url, 'version')