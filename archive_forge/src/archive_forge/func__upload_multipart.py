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
def _upload_multipart(self, response, data, iterator, container, object_name, calculate_hash=True):
    """
        Callback invoked for uploading data to OSS using Aliyun's
        multipart upload mechanism

        :param response: Response object from the initial POST request
        :type response: :class:`OSSRawResponse`

        :param data: Any data from the initial POST request
        :type data: ``str``

        :param iterator: The generator for fetching the upload data
        :type iterator: ``generator``

        :param container: The container owning the object to which data is
            being uploaded
        :type container: :class:`Container`

        :param object_name: The name of the object to which we are uploading
        :type object_name: ``str``

        :keyword calculate_hash: Indicates if we must calculate the data hash
        :type calculate_hash: ``bool``

        :return: A tuple of (status, checksum, bytes transferred)
        :rtype: ``tuple``
        """
    object_path = self._get_object_path(container, object_name)
    response.body = response.response.read()
    body = response.parse_body()
    upload_id = body.find(fixxpath(xpath='UploadId', namespace=self.namespace)).text
    try:
        result = self._upload_from_iterator(iterator, object_path, upload_id, calculate_hash, container=container)
        chunks, data_hash, bytes_transferred = result
        etag = self._commit_multipart(object_path, upload_id, chunks, container=container)
    except Exception as e:
        self._abort_multipart(object_path, upload_id, container=container)
        raise e
    response.headers['etag'] = etag
    return (True, data_hash, bytes_transferred)