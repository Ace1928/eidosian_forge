import os
import re
import binascii
import itertools
from copy import copy
from datetime import datetime
from libcloud.utils.py3 import httplib
from libcloud.compute.base import (
from libcloud.common.linode import (
from libcloud.compute.types import Provider, NodeState, StorageVolumeState
from libcloud.utils.networking import is_private_subnet
class LinodeNodeDriver(NodeDriver):
    name = 'Linode'
    website = 'http://www.linode.com/'
    type = Provider.LINODE

    def __new__(cls, key, secret=None, secure=True, host=None, port=None, api_version=DEFAULT_API_VERSION, region=None, **kwargs):
        if cls is LinodeNodeDriver:
            if api_version == '3.0':
                cls = LinodeNodeDriverV3
            elif api_version == '4.0':
                cls = LinodeNodeDriverV4
            else:
                raise NotImplementedError('No Linode driver found for API version: %s' % api_version)
        return super().__new__(cls)