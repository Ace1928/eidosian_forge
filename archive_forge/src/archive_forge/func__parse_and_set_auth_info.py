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
def _parse_and_set_auth_info(self, data):
    result = {}
    self.account_id = data['accountId']
    self.api_url = data['apiUrl']
    self.download_url = data['downloadUrl']
    self.auth_token = data['authorizationToken']
    parsed_api_url = urlparse.urlparse(self.api_url)
    self.api_host = parsed_api_url.netloc
    parsed_download_url = urlparse.urlparse(self.download_url)
    self.download_host = parsed_download_url.netloc
    return result