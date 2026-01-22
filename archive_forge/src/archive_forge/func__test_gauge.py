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
def _test_gauge(cl, proto):
    cl.gauge('foo', 30)
    _sock_check(cl._sock, 1, proto, 'foo:30|g')
    cl.gauge('foo', 1.2)
    _sock_check(cl._sock, 2, proto, 'foo:1.2|g')
    cl.gauge('foo', 70, rate=0.5)
    _sock_check(cl._sock, 3, proto, 'foo:70|g|@0.5')