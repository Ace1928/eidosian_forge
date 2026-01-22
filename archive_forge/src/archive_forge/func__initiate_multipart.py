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
def _initiate_multipart(self, container, object_name, headers=None):
    """
        Initiates a multipart upload to S3

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :keyword headers: Additional headers to send with the request
        :type headers: ``dict``

        :return: The id of the newly created multipart upload
        :rtype: ``str``
        """
    headers = headers or {}
    request_path = self._get_object_path(container, object_name)
    params = {'uploads': ''}
    response = self.connection.request(request_path, method='POST', headers=headers, params=params)
    if response.status != httplib.OK:
        raise LibcloudError('Error initiating multipart upload', driver=self)
    return findtext(element=response.object, xpath='UploadId', namespace=self.namespace)