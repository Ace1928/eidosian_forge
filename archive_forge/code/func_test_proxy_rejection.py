import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def test_proxy_rejection(self):
    evt = threading.Event()

    def request_handler(listener):
        sock = listener.accept()[0]
        handler = handle_socks4_negotiation(sock)
        addr, port = next(handler)
        handler.send(False)
        evt.wait()
        sock.close()
    self._start_server(request_handler)
    proxy_url = 'socks4a://%s:%s' % (self.host, self.port)
    with socks.SOCKSProxyManager(proxy_url) as pm:
        with pytest.raises(NewConnectionError):
            pm.request('GET', 'http://example.com', retries=False)
        evt.set()