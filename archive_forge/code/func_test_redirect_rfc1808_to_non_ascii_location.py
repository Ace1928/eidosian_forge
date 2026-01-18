import pytest
import threading
import requests
from tests.testserver.server import Server, consume_socket_content
from .utils import override_environ
def test_redirect_rfc1808_to_non_ascii_location():
    path = u'š'
    expected_path = b'%C5%A1'
    redirect_request = []

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = u'//{}:{}/{}'.format(host, port, path)
        sock.send(b'HTTP/1.1 301 Moved Permanently\r\nContent-Length: 0\r\nLocation: ' + location.encode('utf8') + b'\r\n\r\n')
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')
    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)
    with server as (host, port):
        url = u'http://{}:{}'.format(host, port)
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == u'{}/{}'.format(url, expected_path.decode('ascii'))
        close_server.set()