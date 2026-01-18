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
def upload_object_via_stream(self, iterator, container, object_name, extra=None, headers=None, ex_storage_class=None):
    """
        @inherits: :class:`StorageDriver.upload_object_via_stream`

        :param ex_storage_class: Storage class
        :type ex_storage_class: ``str``
        """
    method = 'PUT'
    params = None
    if self.supports_s3_multipart_upload:
        return self._put_object_multipart(container=container, object_name=object_name, extra=extra, stream=iterator, verify_hash=False, headers=headers, storage_class=ex_storage_class)
    return self._put_object(container=container, object_name=object_name, extra=extra, method=method, query_args=params, stream=iterator, verify_hash=False, headers=headers, storage_class=ex_storage_class)