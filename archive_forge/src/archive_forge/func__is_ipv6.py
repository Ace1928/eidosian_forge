from collections import abc
import errno
import socket
import ssl
import warnings
import httplib2
import six.moves.http_client
import urllib3
def _is_ipv6(addr):
    """Checks if a given address is an IPv6 address."""
    try:
        socket.getaddrinfo(addr, None, 0, 0, 0, socket.AI_NUMERICHOST)
    except socket.gaierror:
        return False
    try:
        socket.inet_aton(addr)
        return False
    except socket.error:
        return True