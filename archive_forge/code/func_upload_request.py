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
def upload_request(self, action, headers, upload_host, auth_token, data):
    auth_conn = self._auth_conn.authenticate()
    self._set_host(host=upload_host)
    method = 'POST'
    raw = False
    response = self._request(auth_conn=auth_conn, action=action, params=None, data=data, headers=headers, method=method, raw=raw, auth_token=auth_token)
    return response