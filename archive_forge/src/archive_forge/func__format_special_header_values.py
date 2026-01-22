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
def _format_special_header_values(self, headers, method):
    is_change = method not in ('GET', 'HEAD')
    is_old_api = self.API_VERSION <= '2014-02-14'
    special_header_keys = ['content-encoding', 'content-language', 'content-length', 'content-md5', 'content-type', 'date', 'if-modified-since', 'if-match', 'if-none-match', 'if-unmodified-since', 'range']
    special_header_values = []
    for header in special_header_keys:
        header = header.lower()
        if header in headers:
            special_header_values.append(headers[header])
        elif header == 'content-length' and is_change and is_old_api:
            special_header_values.append('0')
        else:
            special_header_values.append('')
    return special_header_values