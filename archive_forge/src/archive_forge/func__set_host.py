import base64
import hashlib
from libcloud.utils.py3 import b, next, httplib, urlparse
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks, exhaust_iterator
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.utils.escape import sanitize_object_name
from libcloud.storage.types import ObjectDoesNotExistError, ContainerDoesNotExistError
from libcloud.storage.providers import Provider
def _set_host(self, host):
    """
        Dynamically set host which will be used for the following HTTP
        requests.

        NOTE: This is needed because Backblaze uses different hosts for API,
        download and upload requests.
        """
    self.host = host
    self.connection.host = 'https://%s' % host