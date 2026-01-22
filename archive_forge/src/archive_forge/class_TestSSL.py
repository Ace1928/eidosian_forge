from dummyserver.server import (
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.exceptions import (
from urllib3.packages.six.moves import http_client as httplib
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn, onlyPy3
import os
import os.path
import select
import shutil
import socket
import ssl
import sys
import tempfile
from collections import OrderedDict
from test import (
from threading import Event
import mock
import pytest
import trustme
class TestSSL(SocketDummyServerTestCase):

    def test_ssl_failure_midway_through_conn(self):

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock2 = sock.dup()
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            sock2.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nHi'.encode('utf-8'))
            sock2.close()
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port) as pool:
            with pytest.raises(MaxRetryError) as cm:
                pool.request('GET', '/', retries=0)
            assert isinstance(cm.value.reason, SSLError)

    @notSecureTransport
    def test_ssl_read_timeout(self):
        timed_out = Event()

        def socket_handler(listener):
            sock = listener.accept()[0]
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 10\r\n\r\nHi-'.encode('utf-8'))
            timed_out.wait()
            sock.close()
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()

    def test_ssl_failed_fingerprint_verification(self):

        def socket_handler(listener):
            for i in range(2):
                sock = listener.accept()[0]
                ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
                ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
                ssl_sock.close()
                sock.close()
        self._start_server(socket_handler)
        fingerprint = 'A0:C4:A7:46:00:ED:A7:2D:C0:BE:CB:9A:8C:B6:07:CA:58:EE:74:5E'

        def request():
            pool = HTTPSConnectionPool(self.host, self.port, assert_fingerprint=fingerprint)
            try:
                timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                response = pool.urlopen('GET', '/', preload_content=False, retries=0, timeout=timeout)
                response.read()
            finally:
                pool.close()
        with pytest.raises(MaxRetryError) as cm:
            request()
        assert isinstance(cm.value.reason, SSLError)
        with pytest.raises(MaxRetryError):
            request()

    def test_retry_ssl_error(self):

        def socket_handler(listener):
            sock = listener.accept()[0]
            sock2 = sock.dup()
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            sock2.send('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n\r\nFail'.encode('utf-8'))
            sock2.close()
            ssl_sock.close()
            sock = listener.accept()[0]
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 7\r\n\r\nSuccess')
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
            response = pool.urlopen('GET', '/', retries=1)
            assert response.data == b'Success'

    def test_ssl_load_default_certs_when_empty(self):

        def socket_handler(listener):
            sock = listener.accept()[0]
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
            ssl_sock.close()
            sock.close()
        context = mock.create_autospec(ssl_.SSLContext)
        context.load_default_certs = mock.Mock()
        context.options = 0
        with mock.patch('urllib3.util.ssl_.SSLContext', lambda *_, **__: context):
            self._start_server(socket_handler)
            with HTTPSConnectionPool(self.host, self.port) as pool:
                with pytest.raises(MaxRetryError):
                    pool.request('GET', '/', timeout=SHORT_TIMEOUT)
                context.load_default_certs.assert_called_with()

    @notPyPy2
    def test_ssl_dont_load_default_certs_when_given(self):

        def socket_handler(listener):
            sock = listener.accept()[0]
            ssl_sock = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
            ssl_sock.close()
            sock.close()
        context = mock.create_autospec(ssl_.SSLContext)
        context.load_default_certs = mock.Mock()
        context.options = 0
        with mock.patch('urllib3.util.ssl_.SSLContext', lambda *_, **__: context):
            for kwargs in [{'ca_certs': '/a'}, {'ca_cert_dir': '/a'}, {'ca_certs': 'a', 'ca_cert_dir': 'a'}, {'ssl_context': context}]:
                self._start_server(socket_handler)
                with HTTPSConnectionPool(self.host, self.port, **kwargs) as pool:
                    with pytest.raises(MaxRetryError):
                        pool.request('GET', '/', timeout=SHORT_TIMEOUT)
                    context.load_default_certs.assert_not_called()

    def test_load_verify_locations_exception(self):
        """
        Ensure that load_verify_locations raises SSLError for all backends
        """
        with pytest.raises(SSLError):
            ssl_wrap_socket(None, ca_certs='/tmp/fake-file')

    def test_ssl_custom_validation_failure_terminates(self, tmpdir):
        """
        Ensure that the underlying socket is terminated if custom validation fails.
        """
        server_closed = Event()

        def is_closed_socket(sock):
            try:
                sock.settimeout(SHORT_TIMEOUT)
                sock.recv(1)
            except (OSError, socket.error):
                return True
            return False

        def socket_handler(listener):
            sock = listener.accept()[0]
            try:
                _ = ssl.wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            except ssl.SSLError as e:
                assert 'alert unknown ca' in str(e)
                if is_closed_socket(sock):
                    server_closed.set()
        self._start_server(socket_handler)
        other_ca = trustme.CA()
        other_ca_path = str(tmpdir / 'ca.pem')
        other_ca.cert_pem.write_to_path(other_ca_path)
        with HTTPSConnectionPool(self.host, self.port, cert_reqs='REQUIRED', ca_certs=other_ca_path) as pool:
            with pytest.raises(SSLError):
                pool.request('GET', '/', retries=False, timeout=LONG_TIMEOUT)
        assert server_closed.wait(LONG_TIMEOUT), 'The socket was not terminated'