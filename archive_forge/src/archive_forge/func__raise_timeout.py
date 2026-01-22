from __future__ import absolute_import
import errno
import logging
import re
import socket
import sys
import warnings
from socket import error as SocketError
from socket import timeout as SocketTimeout
from .connection import (
from .exceptions import (
from .packages import six
from .packages.six.moves import queue
from .request import RequestMethods
from .response import HTTPResponse
from .util.connection import is_connection_dropped
from .util.proxy import connection_requires_http_tunnel
from .util.queue import LifoQueue
from .util.request import set_file_position
from .util.response import assert_header_parsing
from .util.retry import Retry
from .util.ssl_match_hostname import CertificateError
from .util.timeout import Timeout
from .util.url import Url, _encode_target
from .util.url import _normalize_host as normalize_host
from .util.url import get_host, parse_url
def _raise_timeout(self, err, url, timeout_value):
    """Is the error actually a timeout? Will raise a ReadTimeout or pass"""
    if isinstance(err, SocketTimeout):
        raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
    if hasattr(err, 'errno') and err.errno in _blocking_errnos:
        raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)
    if 'timed out' in str(err) or 'did not complete (read)' in str(err):
        raise ReadTimeoutError(self, url, 'Read timed out. (read timeout=%s)' % timeout_value)