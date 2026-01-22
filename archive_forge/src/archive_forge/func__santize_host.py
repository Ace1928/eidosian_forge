import os
import base64
import warnings
from typing import Optional
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import (
from libcloud.common.types import InvalidCredsError
def _santize_host(self, host=None):
    """
        Sanitize "host" argument any remove any protocol prefix (if specified).
        """
    if not host:
        return None
    prefixes = ['http://', 'https://']
    for prefix in prefixes:
        if host.startswith(prefix):
            host = host.lstrip(prefix)
    return host