import threading
import socket
import time
import pytest
import requests
from tests.testserver.server import Server
def test_request_recovery_with_bigger_timeout(self):
    """a biggest timeout can be specified"""
    server = Server.basic_response_server(request_timeout=3)
    data = b'bananadine'
    with server as address:
        sock = socket.socket()
        sock.connect(address)
        time.sleep(1.5)
        sock.sendall(data)
        sock.close()
    assert server.handler_results[0] == data