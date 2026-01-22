from __future__ import print_function
import email.utils
import errno
import hashlib
import mimetypes
import os
import re
import base64
import binascii
import math
from hashlib import md5
import boto.utils
from boto.compat import BytesIO, six, urllib, encodebytes
from boto.exception import BotoClientError
from boto.exception import StorageDataError
from boto.exception import PleaseRetryException
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.provider import Provider
from boto.s3.keyfile import KeyFile
from boto.s3.user import User
from boto import UserAgent
import boto.utils
from boto.utils import compute_md5, compute_hash
from boto.utils import find_matching_headers
from boto.utils import merge_headers_by_name
from boto.utils import print_to_fd
def _get_storage_class(self):
    if self._storage_class is None and self.bucket:
        list_items = list(self.bucket.list(self.name.encode('utf-8')))
        if len(list_items) and getattr(list_items[0], '_storage_class', None):
            self._storage_class = list_items[0]._storage_class
        else:
            self._storage_class = 'STANDARD'
    return self._storage_class