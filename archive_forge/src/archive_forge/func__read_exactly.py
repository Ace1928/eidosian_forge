import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def _read_exactly(sock, amt):
    """
    Read *exactly* ``amt`` bytes from the socket ``sock``.
    """
    data = b''
    while amt > 0:
        chunk = sock.recv(amt)
        data += chunk
        amt -= len(chunk)
    return data