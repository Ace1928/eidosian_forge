import socket
from .base import StatsClientBase, PipelineBase
class TCPStatsClient(StreamClientBase):
    """TCP version of StatsClient."""

    def __init__(self, host='localhost', port=8125, prefix=None, timeout=None, ipv6=False):
        """Create a new client."""
        self._host = host
        self._port = port
        self._ipv6 = ipv6
        self._timeout = timeout
        self._prefix = prefix
        self._sock = None

    def connect(self):
        fam = socket.AF_INET6 if self._ipv6 else socket.AF_INET
        family, _, _, _, addr = socket.getaddrinfo(self._host, self._port, fam, socket.SOCK_STREAM)[0]
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        self._sock.connect(addr)