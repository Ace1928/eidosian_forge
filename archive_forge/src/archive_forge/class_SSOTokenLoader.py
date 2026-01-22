import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
class SSOTokenLoader:

    def __init__(self, cache=None):
        if cache is None:
            cache = {}
        self._cache = cache

    def _generate_cache_key(self, start_url, session_name):
        input_str = start_url
        if session_name is not None:
            input_str = session_name
        return hashlib.sha1(input_str.encode('utf-8')).hexdigest()

    def save_token(self, start_url, token, session_name=None):
        cache_key = self._generate_cache_key(start_url, session_name)
        self._cache[cache_key] = token

    def __call__(self, start_url, session_name=None):
        cache_key = self._generate_cache_key(start_url, session_name)
        logger.debug(f'Checking for cached token at: {cache_key}')
        if cache_key not in self._cache:
            name = start_url
            if session_name is not None:
                name = session_name
            error_msg = f'Token for {name} does not exist'
            raise SSOTokenLoadError(error_msg=error_msg)
        token = self._cache[cache_key]
        if 'accessToken' not in token or 'expiresAt' not in token:
            error_msg = f'Token for {start_url} is invalid'
            raise SSOTokenLoadError(error_msg=error_msg)
        return token