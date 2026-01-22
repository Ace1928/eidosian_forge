import os
import hmac
import time
import base64
from typing import Dict, Optional
from hashlib import sha1
from datetime import datetime
import libcloud.utils.py3
from libcloud.utils.py3 import b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import findtext, fixxpath
from libcloud.common.aws import (
from libcloud.common.base import RawResponse, ConnectionUserAndKey
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _headers_to_object(self, object_name, container, headers):
    hash = headers.get('etag', '').replace('"', '')
    extra = {}
    if 'content-type' in headers:
        extra['content_type'] = headers['content-type']
    if 'etag' in headers:
        extra['etag'] = headers['etag']
    meta_data = {}
    if 'content-encoding' in headers:
        extra['content_encoding'] = headers['content-encoding']
    if 'last-modified' in headers:
        extra['last_modified'] = headers['last-modified']
    for key, value in headers.items():
        if not key.lower().startswith(self.http_vendor_prefix + '-meta-'):
            continue
        key = key.replace(self.http_vendor_prefix + '-meta-', '')
        meta_data[key] = value
    content_length = self._get_content_length_from_headers(headers=headers)
    if content_length is None:
        raise KeyError('Can not deduce object size from headers for object %s' % object_name)
    obj = Object(name=object_name, size=int(content_length), hash=hash or None, extra=extra, meta_data=meta_data, container=container, driver=self)
    return obj