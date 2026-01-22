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
class S3MultipartUpload:
    """
    Class representing an amazon s3 multipart upload
    """

    def __init__(self, key, id, created_at, initiator, owner):
        """
        Class representing an amazon s3 multipart upload

        :param key: The object/key that was being uploaded
        :type key: ``str``

        :param id: The upload id assigned by amazon
        :type id: ``str``

        :param created_at: The date/time at which the upload was started
        :type created_at: ``str``

        :param initiator: The AWS owner/IAM user who initiated this
        :type initiator: ``str``

        :param owner: The AWS owner/IAM who will own this object
        :type owner: ``str``
        """
        self.key = key
        self.id = id
        self.created_at = created_at
        self.initiator = initiator
        self.owner = owner

    def __repr__(self):
        return '<S3MultipartUpload: key=%s>' % self.key