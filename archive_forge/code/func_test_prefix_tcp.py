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
def test_prefix_tcp():
    """TCPStatsClient.incr works."""
    cl = _tcp_client(prefix='foo')
    _test_prefix(cl, 'tcp')