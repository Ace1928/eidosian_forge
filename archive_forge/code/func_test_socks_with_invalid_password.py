import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_socks_with_invalid_password(self):

    def request_handler(listener):
        sock = listener.accept()[0]
        handler = handle_socks5_negotiation(sock, negotiate=True, username=b'user', password=b'pass')
        next(handler)
    self._start_server(request_handler)
    proxy_url = 'socks5h://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url, username='user', password='badpass') as pm:
        with pytest.raises(NewConnectionError) as e:
            pm.request('GET', 'http://example.com', retries=False)
        assert 'SOCKS5 authentication failed' in str(e.value)