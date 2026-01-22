import os
import hmac
import atexit
from time import time
from hashlib import sha1
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import Response, RawResponse
from libcloud.utils.files import read_in_chunks
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.storage.base import Object, Container, StorageDriver
from libcloud.storage.types import (
from libcloud.common.openstack import OpenStackDriverMixin, OpenStackBaseConnection
from libcloud.common.rackspace import AUTH_URL
from libcloud.storage.providers import Provider
from io import FileIO as file
def _upload_object_part(self, container, object_name, part_number, iterator, verify_hash=True):
    part_name = object_name + '/%08d' % part_number
    extra = {'content_type': 'application/octet-stream'}
    self._put_object(container=container, object_name=part_name, extra=extra, stream=iterator, verify_hash=verify_hash)