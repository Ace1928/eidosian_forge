import socket
import threading
from test import SHORT_TIMEOUT
import pytest
from dummyserver.server import DEFAULT_CA, DEFAULT_CERTS
from dummyserver.testcase import IPV4SocketDummyServerTestCase
from urllib3.contrib import socks
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
def _address_from_socket(sock):
    """
    Returns the address from the SOCKS socket
    """
    addr_type = sock.recv(1)
    if addr_type == b'\x01':
        ipv4_addr = _read_exactly(sock, 4)
        return socket.inet_ntoa(ipv4_addr)
    elif addr_type == b'\x04':
        ipv6_addr = _read_exactly(sock, 16)
        return socket.inet_ntop(socket.AF_INET6, ipv6_addr)
    elif addr_type == b'\x03':
        addr_len = ord(sock.recv(1))
        return _read_exactly(sock, addr_len)
    else:
        raise RuntimeError('Unexpected addr type: %r' % addr_type)