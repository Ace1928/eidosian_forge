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
class JSONFileCache:
    """JSON file cache.
    This provides a dict like interface that stores JSON serializable
    objects.
    The objects are serialized to JSON and stored in a file.  These
    values can be retrieved at a later time.
    """
    CACHE_DIR = os.path.expanduser(os.path.join('~', '.aws', 'boto', 'cache'))

    def __init__(self, working_dir=CACHE_DIR, dumps_func=None):
        self._working_dir = working_dir
        if dumps_func is None:
            dumps_func = self._default_dumps
        self._dumps = dumps_func

    def _default_dumps(self, obj):
        return json.dumps(obj, default=self._serialize_if_needed)

    def __contains__(self, cache_key):
        actual_key = self._convert_cache_key(cache_key)
        return os.path.isfile(actual_key)

    def __getitem__(self, cache_key):
        """Retrieve value from a cache key."""
        actual_key = self._convert_cache_key(cache_key)
        try:
            with open(actual_key) as f:
                return json.load(f)
        except (OSError, ValueError):
            raise KeyError(cache_key)

    def __delitem__(self, cache_key):
        actual_key = self._convert_cache_key(cache_key)
        try:
            key_path = Path(actual_key)
            key_path.unlink()
        except FileNotFoundError:
            raise KeyError(cache_key)

    def __setitem__(self, cache_key, value):
        full_key = self._convert_cache_key(cache_key)
        try:
            file_content = self._dumps(value)
        except (TypeError, ValueError):
            raise ValueError(f'Value cannot be cached, must be JSON serializable: {value}')
        if not os.path.isdir(self._working_dir):
            os.makedirs(self._working_dir)
        with os.fdopen(os.open(full_key, os.O_WRONLY | os.O_CREAT, 384), 'w') as f:
            f.truncate()
            f.write(file_content)

    def _convert_cache_key(self, cache_key):
        full_path = os.path.join(self._working_dir, cache_key + '.json')
        return full_path

    def _serialize_if_needed(self, value, iso=False):
        if isinstance(value, _DatetimeClass):
            if iso:
                return value.isoformat()
            return value.strftime('%Y-%m-%dT%H:%M:%S%Z')
        return value