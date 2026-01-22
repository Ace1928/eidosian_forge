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
def _encode_container_name(self, name):
    """
        Encode container name so it can be used as part of the HTTP request.
        """
    if name.startswith('/'):
        name = name[1:]
    name = urlquote(name)
    if name.find('/') != -1:
        raise InvalidContainerNameError(value='Container name cannot contain slashes', container_name=name, driver=self)
    if len(name) > 256:
        raise InvalidContainerNameError(value='Container name cannot be longer than 256 bytes', container_name=name, driver=self)
    return name