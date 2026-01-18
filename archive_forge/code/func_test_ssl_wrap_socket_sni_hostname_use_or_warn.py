import hashlib
import io
import logging
import socket
import ssl
import warnings
from itertools import chain
from test import notBrotlipy, onlyBrotlipy, onlyPy2, onlyPy3
import pytest
from mock import Mock, patch
from urllib3 import add_stderr_logger, disable_warnings, util
from urllib3.exceptions import (
from urllib3.packages import six
from urllib3.poolmanager import ProxyConfig
from urllib3.util import is_fp_closed
from urllib3.util.connection import _has_ipv6, allowed_gai_family, create_connection
from urllib3.util.proxy import connection_requires_http_tunnel, create_proxy_ssl_context
from urllib3.util.request import _FAILEDTELL, make_headers, rewind_body
from urllib3.util.response import assert_header_parsing
from urllib3.util.ssl_ import (
from urllib3.util.timeout import Timeout
from urllib3.util.url import Url, get_host, parse_url, split_first
from . import clear_warnings
def test_ssl_wrap_socket_sni_hostname_use_or_warn(self):
    """Test that either an SNI hostname is used or a warning is made."""
    sock = object()
    context, warn = self._wrap_socket_and_mock_warn(sock, 'www.google.com')
    if util.HAS_SNI:
        warn.assert_not_called()
        context.wrap_socket.assert_called_once_with(sock, server_hostname='www.google.com')
    else:
        assert warn.call_count >= 1
        warnings = [call[0][1] for call in warn.call_args_list]
        assert SNIMissingWarning in warnings
        context.wrap_socket.assert_called_once_with(sock)