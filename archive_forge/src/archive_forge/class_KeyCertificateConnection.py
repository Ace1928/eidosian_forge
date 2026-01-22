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
class KeyCertificateConnection(CertificateConnection):
    """
    Base connection class which accepts both ``key_file`` and ``cert_file``
    argument.
    """
    key_file = None

    def __init__(self, key_file, cert_file, secure=True, host=None, port=None, url=None, proxy_url=None, timeout=None, backoff=None, retry_delay=None, ca_cert=None):
        """
        Initialize `cert_file`; set `secure` to an ``int`` based on
        passed value.
        """
        super().__init__(cert_file, secure=secure, host=host, port=port, url=url, timeout=timeout, backoff=backoff, retry_delay=retry_delay, proxy_url=proxy_url)
        self.key_file = key_file