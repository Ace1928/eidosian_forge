from libcloud.common.types import LibcloudError
from libcloud.storage.providers import Provider
from libcloud.storage.drivers.cloudfiles import CloudFilesConnection, CloudFilesStorageDriver
class KTUCloudStorageConnection(CloudFilesConnection):
    """
    Connection class for the KT UCloud Storage endpoint.
    """
    auth_url = KTUCLOUDSTORAGE_AUTH_URL
    _auth_version = KTUCLOUDSTORAGE_API_VERSION

    def get_endpoint(self):
        eps = self.service_catalog.get_endpoints(name='cloudFiles')
        if len(eps) == 0:
            raise LibcloudError('Could not find specified endpoint')
        ep = eps[0]
        public_url = ep.url
        if not public_url:
            raise LibcloudError('Could not find specified endpoint')
        return public_url