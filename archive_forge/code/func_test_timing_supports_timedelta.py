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
def test_timing_supports_timedelta():
    cl = _udp_client()
    proto = 'udp'
    cl.timing('foo', timedelta(seconds=1.5))
    _sock_check(cl._sock, 1, proto, 'foo:1500.000000|ms')
    cl.timing('foo', timedelta(days=1.5))
    _sock_check(cl._sock, 2, proto, 'foo:129600000.000000|ms')