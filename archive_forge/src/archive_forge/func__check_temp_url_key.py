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
def _check_temp_url_key(self, container=None, temp_url_key=None):
    if temp_url_key:
        if not isinstance(temp_url_key, bytes):
            temp_url_key = temp_url_key.encode('utf8')
    else:
        temp_url_key = self.get_temp_url_key(container)
    if not temp_url_key:
        raise exceptions.SDKException('temp_url_key was not given, nor was a temporary url key found for the account or the container.')
    return temp_url_key