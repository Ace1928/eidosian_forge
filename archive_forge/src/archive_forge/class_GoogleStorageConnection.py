import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
class GoogleStorageConnection(ConnectionUserAndKey):
    """
    Represents a single connection to the Google storage API endpoint.

    This can either authenticate via the Google OAuth2 methods or via
    the S3 HMAC interoperability method.
    """
    host = 'storage.googleapis.com'
    responseCls = S3Response
    rawResponseCls = S3RawResponse
    PROJECT_ID_HEADER = 'x-goog-project-id'

    def __init__(self, user_id, key, secure=True, auth_type=None, credential_file=None, **kwargs):
        self.auth_type = auth_type or GoogleAuthType.guess_type(user_id)
        if GoogleAuthType.is_oauth2(self.auth_type):
            self.oauth2_credential = GoogleOAuth2Credential(user_id, key, self.auth_type, credential_file, **kwargs)
        else:
            self.oauth2_credential = None
        super().__init__(user_id, key, secure, **kwargs)

    def add_default_headers(self, headers):
        date = email.utils.formatdate(usegmt=True)
        headers['Date'] = date
        project = self.get_project()
        if project:
            headers[self.PROJECT_ID_HEADER] = project
        return headers

    def get_project(self):
        return getattr(self.driver, 'project', None)

    def pre_connect_hook(self, params, headers):
        if self.auth_type == GoogleAuthType.GCS_S3:
            signature = self._get_s3_auth_signature(params, headers)
            headers['Authorization'] = '{} {}:{}'.format(SIGNATURE_IDENTIFIER, self.user_id, signature)
        else:
            headers['Authorization'] = 'Bearer ' + self.oauth2_credential.access_token
        return (params, headers)

    def _get_s3_auth_signature(self, params, headers):
        """Hacky wrapper to work with S3's get_auth_signature."""
        headers_copy = {}
        params_copy = copy.deepcopy(params)
        for k, v in headers.items():
            k_lower = k.lower()
            if k_lower in ['date', 'content-type'] or k_lower.startswith(GoogleStorageDriver.http_vendor_prefix) or (not isinstance(v, str)):
                headers_copy[k_lower] = v
            else:
                headers_copy[k_lower] = v.lower()
        return BaseS3Connection.get_auth_signature(method=self.method, headers=headers_copy, params=params_copy, expires=None, secret_key=self.key, path=self.action, vendor_prefix=GoogleStorageDriver.http_vendor_prefix)