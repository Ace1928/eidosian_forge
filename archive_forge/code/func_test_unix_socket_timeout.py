import asyncio
import functools
import random
import re
import socket
from datetime import timedelta
from unittest import SkipTest, mock
from statsd import StatsClient
from statsd import TCPStatsClient
from statsd import UnixSocketStatsClient
@mock.patch.object(socket, 'socket')
def test_unix_socket_timeout(mock_socket):
    """Timeout on UnixSocketStatsClient should be set on socket."""
    test_timeout = 321
    cl = UnixSocketStatsClient(UNIX_SOCKET, timeout=test_timeout)
    cl.incr('foo')
    cl._sock.settimeout.assert_called_once_with(test_timeout)