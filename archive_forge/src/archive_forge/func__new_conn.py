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
def _new_conn(self):
    """
        Return a fresh :class:`http.client.HTTPSConnection`.
        """
    self.num_connections += 1
    log.debug('Starting new HTTPS connection (%d): %s:%s', self.num_connections, self.host, self.port or '443')
    if not self.ConnectionCls or self.ConnectionCls is DummyConnection:
        raise SSLError("Can't connect to HTTPS URL because the SSL module is not available.")
    actual_host = self.host
    actual_port = self.port
    if self.proxy is not None:
        actual_host = self.proxy.host
        actual_port = self.proxy.port
    conn = self.ConnectionCls(host=actual_host, port=actual_port, timeout=self.timeout.connect_timeout, strict=self.strict, cert_file=self.cert_file, key_file=self.key_file, key_password=self.key_password, **self.conn_kw)
    return self._prepare_conn(conn)