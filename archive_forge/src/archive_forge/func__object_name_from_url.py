from calendar import timegm
import collections
from hashlib import sha1
import hmac
import json
import os
import time
from urllib import parse
from openstack import _log
from openstack.cloud import _utils
from openstack import exceptions
from openstack.object_store.v1 import account as _account
from openstack.object_store.v1 import container as _container
from openstack.object_store.v1 import info as _info
from openstack.object_store.v1 import obj as _obj
from openstack import proxy
from openstack import utils
def _object_name_from_url(self, url):
    """Get container_name/object_name from the full URL called.

        Remove the Swift endpoint from the front of the URL, and remove
        the leaving / that will leave behind."""
    endpoint = self.get_endpoint()
    object_name = url.replace(endpoint, '')
    if object_name.startswith('/'):
        object_name = object_name[1:]
    return object_name