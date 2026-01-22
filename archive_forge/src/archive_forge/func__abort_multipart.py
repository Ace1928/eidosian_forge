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
def _abort_multipart(self, container, object_name, upload_id):
    """
        Aborts an already initiated multipart upload

        :param container: The destination container
        :type container: :class:`Container`

        :param object_name: The name of the object which we are uploading
        :type object_name: ``str``

        :param upload_id: The upload id allocated for this multipart upload
        :type upload_id: ``str``
        """
    params = {'uploadId': upload_id}
    request_path = self._get_object_path(container, object_name)
    resp = self.connection.request(request_path, method='DELETE', params=params)
    if resp.status != httplib.NO_CONTENT:
        raise LibcloudError('Error in multipart abort. status_code=%d' % resp.status, driver=self)