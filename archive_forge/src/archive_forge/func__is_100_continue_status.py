import functools
import logging
from collections.abc import Mapping
import urllib3.util
from urllib3.connection import HTTPConnection, VerifiedHTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
import botocore.utils
from botocore.compat import (
from botocore.exceptions import UnseekableStreamError
def _is_100_continue_status(self, maybe_status_line):
    parts = maybe_status_line.split(None, 2)
    return len(parts) >= 3 and parts[0].startswith(b'HTTP/') and (parts[1] == b'100')