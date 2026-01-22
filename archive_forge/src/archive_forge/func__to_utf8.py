import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def _to_utf8(self, item):
    key, value = item
    if isinstance(key, str):
        key = key.encode('utf-8')
    if isinstance(value, str):
        value = value.encode('utf-8')
    return (key, value)