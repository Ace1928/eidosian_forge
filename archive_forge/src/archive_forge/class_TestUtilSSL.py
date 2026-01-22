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
class TestUtilSSL(object):
    """Test utils that use an SSL backend."""

    @pytest.mark.parametrize('candidate, requirements', [(None, ssl.CERT_REQUIRED), (ssl.CERT_NONE, ssl.CERT_NONE), (ssl.CERT_REQUIRED, ssl.CERT_REQUIRED), ('REQUIRED', ssl.CERT_REQUIRED), ('CERT_REQUIRED', ssl.CERT_REQUIRED)])
    def test_resolve_cert_reqs(self, candidate, requirements):
        assert resolve_cert_reqs(candidate) == requirements

    @pytest.mark.parametrize('candidate, version', [(ssl.PROTOCOL_TLSv1, ssl.PROTOCOL_TLSv1), ('PROTOCOL_TLSv1', ssl.PROTOCOL_TLSv1), ('TLSv1', ssl.PROTOCOL_TLSv1), (ssl.PROTOCOL_SSLv23, ssl.PROTOCOL_SSLv23)])
    def test_resolve_ssl_version(self, candidate, version):
        assert resolve_ssl_version(candidate) == version

    def test_ssl_wrap_socket_loads_the_cert_chain(self):
        socket = object()
        mock_context = Mock()
        ssl_wrap_socket(ssl_context=mock_context, sock=socket, certfile='/path/to/certfile')
        mock_context.load_cert_chain.assert_called_once_with('/path/to/certfile', None)

    @patch('urllib3.util.ssl_.create_urllib3_context')
    def test_ssl_wrap_socket_creates_new_context(self, create_urllib3_context):
        socket = object()
        ssl_wrap_socket(sock=socket, cert_reqs='CERT_REQUIRED')
        create_urllib3_context.assert_called_once_with(None, 'CERT_REQUIRED', ciphers=None)

    def test_ssl_wrap_socket_loads_verify_locations(self):
        socket = object()
        mock_context = Mock()
        ssl_wrap_socket(ssl_context=mock_context, ca_certs='/path/to/pem', sock=socket)
        mock_context.load_verify_locations.assert_called_once_with('/path/to/pem', None, None)

    def test_ssl_wrap_socket_loads_certificate_directories(self):
        socket = object()
        mock_context = Mock()
        ssl_wrap_socket(ssl_context=mock_context, ca_cert_dir='/path/to/pems', sock=socket)
        mock_context.load_verify_locations.assert_called_once_with(None, '/path/to/pems', None)

    def test_ssl_wrap_socket_loads_certificate_data(self):
        socket = object()
        mock_context = Mock()
        ssl_wrap_socket(ssl_context=mock_context, ca_cert_data='TOTALLY PEM DATA', sock=socket)
        mock_context.load_verify_locations.assert_called_once_with(None, None, 'TOTALLY PEM DATA')

    def _wrap_socket_and_mock_warn(self, sock, server_hostname):
        mock_context = Mock()
        with patch('warnings.warn') as warn:
            ssl_wrap_socket(ssl_context=mock_context, sock=sock, server_hostname=server_hostname)
        return (mock_context, warn)

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

    def test_ssl_wrap_socket_sni_ip_address_no_warn(self):
        """Test that a warning is not made if server_hostname is an IP address."""
        sock = object()
        context, warn = self._wrap_socket_and_mock_warn(sock, '8.8.8.8')
        if util.IS_SECURETRANSPORT:
            context.wrap_socket.assert_called_once_with(sock, server_hostname='8.8.8.8')
        else:
            context.wrap_socket.assert_called_once_with(sock)
        warn.assert_not_called()

    def test_ssl_wrap_socket_sni_none_no_warn(self):
        """Test that a warning is not made if server_hostname is not given."""
        sock = object()
        context, warn = self._wrap_socket_and_mock_warn(sock, None)
        context.wrap_socket.assert_called_once_with(sock)
        warn.assert_not_called()