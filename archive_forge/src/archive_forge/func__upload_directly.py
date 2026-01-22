import os
import hmac
import base64
import hashlib
import binascii
from datetime import datetime, timedelta
from libcloud.utils.py3 import ET, b, httplib, tostring, urlquote, urlencode
from libcloud.utils.xml import fixxpath
from libcloud.utils.files import read_in_chunks
from libcloud.common.azure import AzureConnection, AzureActiveDirectoryConnection
from libcloud.common.types import LibcloudError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
def _upload_directly(self, stream, object_path, lease, blob_size, meta_data, content_type, object_name, file_path, headers):
    headers = headers or {}
    lease.update_headers(headers)
    self._update_metadata(headers, meta_data)
    headers['Content-Length'] = str(blob_size)
    headers['x-ms-blob-type'] = 'BlockBlob'
    return self._upload_object(object_name=object_name, file_path=file_path, content_type=content_type, request_path=object_path, stream=stream, headers=headers)