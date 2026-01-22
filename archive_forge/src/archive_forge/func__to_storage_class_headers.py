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
def _to_storage_class_headers(self, storage_class):
    """
        Generates request headers given a storage class name.

        :keyword storage_class: The name of the S3 object's storage class
        :type extra: ``str``

        :return: Headers to include in a request
        :rtype: :dict:
        """
    headers = {}
    storage_class = storage_class or 'standard'
    if storage_class not in ['standard', 'reduced_redundancy', 'standard_ia', 'onezone_ia', 'intelligent_tiering', 'glacier', 'deep_archive', 'glacier_ir']:
        raise ValueError('Invalid storage class value: %s' % storage_class)
    key = self.http_vendor_prefix + '-storage-class'
    headers[key] = storage_class.upper()
    return headers