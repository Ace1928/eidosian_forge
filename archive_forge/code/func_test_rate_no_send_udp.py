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
@mock.patch.object(random, 'random', lambda: 2)
def test_rate_no_send_udp():
    """Rate below random value prevents sending with StatsClient.incr."""
    cl = _udp_client()
    _test_rate_no_send(cl, 'udp')