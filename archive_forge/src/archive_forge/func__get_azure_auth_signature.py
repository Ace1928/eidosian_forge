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
def _get_azure_auth_signature(self, method, headers, params, account, secret_key, path='/'):
    """
        Signature = Base64( HMAC-SHA1( YourSecretAccessKeyID,
                            UTF-8-Encoding-Of( StringToSign ) ) ) );

        StringToSign = HTTP-VERB + "
" +
            Content-Encoding + "
" +
            Content-Language + "
" +
            Content-Length + "
" +
            Content-MD5 + "
" +
            Content-Type + "
" +
            Date + "
" +
            If-Modified-Since + "
" +
            If-Match + "
" +
            If-None-Match + "
" +
            If-Unmodified-Since + "
" +
            Range + "
" +
            CanonicalizedHeaders +
            CanonicalizedResource;
        """
    xms_header_values = []
    param_list = []
    headers_copy = {}
    for header, value in headers.items():
        header = header.lower()
        value = str(value).strip()
        if header.startswith('x-ms-'):
            xms_header_values.append((header, value))
        else:
            headers_copy[header] = value
    special_header_values = self._format_special_header_values(headers_copy, method)
    values_to_sign = [method] + special_header_values
    xms_header_values.sort()
    for header, value in xms_header_values:
        values_to_sign.append('{}:{}'.format(header, value))
    values_to_sign.append('/{}{}'.format(account, path))
    for key, value in params.items():
        param_list.append((key.lower(), str(value).strip()))
    param_list.sort()
    for key, value in param_list:
        values_to_sign.append('{}:{}'.format(key, value))
    string_to_sign = b('\n'.join(values_to_sign))
    secret_key = b(secret_key)
    b64_hmac = base64.b64encode(hmac.new(secret_key, string_to_sign, digestmod=sha256).digest())
    return 'SharedKey {}:{}'.format(self.user_id, b64_hmac.decode('utf-8'))