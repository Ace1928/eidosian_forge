import os
import copy
import hmac
import time
import base64
from hashlib import sha256
from libcloud.http import LibcloudConnection
from libcloud.utils.py3 import ET, b, httplib, urlparse, urlencode, basestring
from libcloud.utils.xml import fixxpath
from libcloud.common.base import (
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.common.azure_arm import AzureAuthJsonResponse, publicEnvironments
class AzureActiveDirectoryConnection(ConnectionUserAndKey):
    """
    Represents a single connection to Azure using Azure AD for Blob
    """
    conn_class = LibcloudConnection
    driver = AzureBaseDriver
    name = 'Azure AD Auth'
    responseCls = AzureResponse
    rawResponseCls = AzureRawResponse
    API_VERSION = '2017-11-09'

    def __init__(self, key, secret, secure=True, host=None, port=None, tenant_id=None, identity=None, cloud_environment='default', **kwargs):
        super().__init__(identity, secret, **kwargs)
        if isinstance(cloud_environment, basestring):
            cloud_environment = publicEnvironments[cloud_environment]
        if not isinstance(cloud_environment, dict):
            raise Exception("cloud_environment must be one of '%s' or a dict containing keys 'resourceManagerEndpointUrl', 'activeDirectoryEndpointUrl', 'activeDirectoryResourceId', 'storageEndpointSuffix'" % "', '".join(publicEnvironments.keys()))
        self.login_host = urlparse.urlparse(cloud_environment['activeDirectoryEndpointUrl']).hostname
        self.login_resource = cloud_environment['activeDirectoryResourceId']
        self.host = host
        self.identity = identity
        self.tenant_id = tenant_id
        self.storage_account_id = key

    def add_default_headers(self, headers):
        headers['x-ms-date'] = time.strftime(AZURE_TIME_FORMAT, time.gmtime())
        headers['x-ms-version'] = self.API_VERSION
        headers['Content-Type'] = 'application/xml'
        headers['Authorization'] = 'Bearer %s' % self.access_token
        return headers

    def get_client_credentials(self):
        """
        Log in and get bearer token used to authorize API requests.
        """
        conn = self.conn_class(self.login_host, 443, timeout=self.timeout)
        conn.connect()
        params = urlencode({'grant_type': 'client_credentials', 'client_id': self.user_id, 'client_secret': self.key, 'resource': 'https://storage.azure.com/'})
        headers = {'Content-type': 'application/x-www-form-urlencoded'}
        conn.request('POST', '/%s/oauth2/token' % self.tenant_id, params, headers)
        js = AzureAuthJsonResponse(conn.getresponse(), conn)
        self.access_token = js.object['access_token']
        self.expires_on = js.object['expires_on']

    def connect(self, **kwargs):
        self.get_client_credentials()
        return super().connect(**kwargs)

    def request(self, action, params=None, data=None, headers=None, method='GET', raw=False, stream=False, json=None, retry_failed=None, *kwargs):
        if time.time() + 300 >= int(self.expires_on):
            self.get_client_credentials()
        return super().request(action, params=params, data=data, headers=headers, method=method, raw=raw, stream=stream, json=json, retry_failed=retry_failed)