import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
class KubernetesTokenAuthConnection(ConnectionKey):
    responseCls = KubernetesResponse
    timeout = 60

    def add_default_headers(self, headers):
        if 'Content-Type' not in headers:
            headers['Content-Type'] = 'application/json'
        if self.key:
            headers['Authorization'] = 'Bearer ' + self.key
        else:
            raise ValueError('Please provide a valid token in the key param')
        return headers