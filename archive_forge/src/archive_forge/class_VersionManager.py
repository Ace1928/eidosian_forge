import urllib
from zunclient.common import base
class VersionManager(base.Manager):
    resource_class = Version

    def list(self):
        endpoint = self.api.get_endpoint()
        url = urllib.parse.urlparse(endpoint)
        if url.path.endswith('v1') or '/v1/' in url.path:
            path = url.path[:url.path.rfind('/v1')]
            version_url = '%s://%s%s' % (url.scheme, url.netloc, path)
        else:
            version_url = endpoint
        return self._list(version_url, 'versions')

    def get_current(self):
        for version in self.list():
            if version.status == 'CURRENT':
                return version
        return None