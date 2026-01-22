import re
import copy
import time
import base64
import hashlib
from libcloud.utils.py3 import b, httplib
from libcloud.utils.misc import dict2str, str2list, str2dicts, get_secure_random_string
from libcloud.common.base import Response, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError, InvalidCredsError
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, is_private_subnet
from libcloud.compute.types import Provider, NodeState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.cloudsigma import (
def _to_drive(self, data):
    id = data['uuid']
    name = data['name']
    size = data['size'] / 1024 / 1024 / 1024
    media = data['media']
    status = data['status']
    extra = {'mounted_on': data.get('mounted_on', []), 'storage_type': data.get('storage_type', ''), 'distribution': data['meta'].get('distribution', ''), 'version': data['meta'].get('version', ''), 'os': data['meta'].get('os', ''), 'paid': data['meta'].get('paid', ''), 'architecture': data['meta'].get('arch', ''), 'created_at': data['meta'].get('created_at', '')}
    drive = CloudSigmaDrive(id=id, name=name, size=size, media=media, status=status, driver=self, extra=extra)
    return drive