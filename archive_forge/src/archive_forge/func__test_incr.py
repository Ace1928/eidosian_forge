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
def _test_incr(cl, proto):
    cl.incr('foo')
    _sock_check(cl._sock, 1, proto, val='foo:1|c')
    cl.incr('foo', 10)
    _sock_check(cl._sock, 2, proto, val='foo:10|c')
    cl.incr('foo', 1.2)
    _sock_check(cl._sock, 3, proto, val='foo:1.2|c')
    cl.incr('foo', 10, rate=0.5)
    _sock_check(cl._sock, 4, proto, val='foo:10|c|@0.5')