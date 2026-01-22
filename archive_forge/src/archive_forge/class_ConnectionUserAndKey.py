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
class ConnectionUserAndKey(ConnectionKey):
    """
    Base connection class which accepts a ``user_id`` and ``key`` argument.
    """
    user_id = None

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, backoff=None, retry_delay=None):
        super().__init__(key, secure=secure, host=host, port=port, url=url, timeout=timeout, backoff=backoff, retry_delay=retry_delay, proxy_url=proxy_url)
        self.user_id = user_id