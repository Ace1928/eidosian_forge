import os
import hmac
import time
import base64
import codecs
from hashlib import sha1
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
@staticmethod
def _get_auth_signature(method, headers, params, expires, secret_key, path, vendor_prefix):
    """
        Signature = base64(hmac-sha1(AccessKeySecret,
          VERB + "
"
          + CONTENT-MD5 + "
"
          + CONTENT-TYPE + "
"
          + EXPIRES + "
"
          + CanonicalizedOSSHeaders
          + CanonicalizedResource))
        """
    special_headers = {'content-md5': '', 'content-type': '', 'expires': ''}
    vendor_headers = {}
    for key, value in list(headers.items()):
        key_lower = key.lower()
        if key_lower in special_headers:
            special_headers[key_lower] = value.strip()
        elif key_lower.startswith(vendor_prefix):
            vendor_headers[key_lower] = value.strip()
    if expires:
        special_headers['expires'] = str(expires)
    buf = [method]
    for _, value in sorted(special_headers.items()):
        buf.append(value)
    string_to_sign = '\n'.join(buf)
    buf = []
    for key, value in sorted(vendor_headers.items()):
        buf.append('{}:{}'.format(key, value))
    header_string = '\n'.join(buf)
    values_to_sign = []
    for value in [string_to_sign, header_string, path]:
        if value:
            values_to_sign.append(value)
    string_to_sign = '\n'.join(values_to_sign)
    b64_hmac = base64.b64encode(hmac.new(b(secret_key), b(string_to_sign), digestmod=sha1).digest())
    return b64_hmac