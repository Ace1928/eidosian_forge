import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
class CloudFilesConnection(OpenStackSwiftConnection):
    """
    Base connection class for the Cloudfiles driver.
    """
    responseCls = CloudFilesResponse
    rawResponseCls = CloudFilesRawResponse
    auth_url = AUTH_URL
    _auth_version = '2.0'

    def __init__(self, user_id, key, secure=True, use_internal_url=False, **kwargs):
        super().__init__(user_id, key, secure=secure, **kwargs)
        self.api_version = API_VERSION
        self.accept_format = 'application/json'
        self.cdn_request = False
        self.use_internal_url = use_internal_url

    def get_endpoint(self):
        region = self._ex_force_service_region.upper()
        if '2.0' in self._auth_version:
            ep = self.service_catalog.get_endpoint(service_type='object-store', name='cloudFiles', region=region, endpoint_type='internal' if self.use_internal_url else 'external')
            cdn_ep = self.service_catalog.get_endpoint(service_type='rax:object-cdn', name='cloudFilesCDN', region=region, endpoint_type='external')
        else:
            raise LibcloudError('Auth version "%s" not supported' % self._auth_version)
        if self.cdn_request:
            ep = cdn_ep
        if not ep or not ep.url:
            raise LibcloudError('Could not find specified endpoint')
        return ep.url

    def request(self, action, params=None, data='', headers=None, method='GET', raw=False, cdn_request=False):
        if not headers:
            headers = {}
        if not params:
            params = {}
        self.cdn_request = cdn_request
        params['format'] = 'json'
        if method in ['POST', 'PUT'] and 'Content-Type' not in headers:
            headers.update({'Content-Type': 'application/json; charset=UTF-8'})
        return super().request(action=action, params=params, data=data, method=method, headers=headers, raw=raw, cdn_request=cdn_request)