import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def _add_cache_busting_to_params(self, params):
    """
        Add cache busting parameter to the query parameters of a GET request.

        Parameters are only added if "cache_busting" class attribute is set to
        True.

        Note: This should only be used with *naughty* providers which use
        excessive caching of responses.
        """
    cache_busting_value = binascii.hexlify(os.urandom(8)).decode('ascii')
    if isinstance(params, dict):
        params['cache-busting'] = cache_busting_value
    else:
        params.append(('cache-busting', cache_busting_value))
    return params