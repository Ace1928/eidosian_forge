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
def _test_timing(cl, proto):
    cl.timing('foo', 100)
    _sock_check(cl._sock, 1, proto, 'foo:100.000000|ms')
    cl.timing('foo', 350)
    _sock_check(cl._sock, 2, proto, 'foo:350.000000|ms')
    cl.timing('foo', 100, rate=0.5)
    _sock_check(cl._sock, 3, proto, 'foo:100.000000|ms|@0.5')