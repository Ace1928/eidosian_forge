import copy
import json
import email.utils
from typing import Dict, Optional
from libcloud.utils.py3 import httplib, urlquote
from libcloud.common.base import ConnectionUserAndKey
from libcloud.common.types import ProviderError
from libcloud.common.google import GoogleAuthType, GoogleResponse, GoogleOAuth2Credential
from libcloud.storage.drivers.s3 import (
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