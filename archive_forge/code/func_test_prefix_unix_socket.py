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
@mock.patch.object(random, 'random', lambda: -1)
def test_prefix_unix_socket():
    """UnixSocketStatsClient.incr works."""
    cl = _unix_socket_client(prefix='foo')
    _test_prefix(cl, 'unix')