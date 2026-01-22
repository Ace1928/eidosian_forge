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
def _test_timer_decorator(cl, proto):

    @cl.timer('foo')
    def foo(a, b):
        return [a, b]

    @cl.timer('bar')
    def bar(a, b):
        return [b, a]
    eq_([4, 2], foo(4, 2))
    _timer_check(cl._sock, 1, proto, 'foo', 'ms')
    eq_([2, 4], bar(4, 2))
    _timer_check(cl._sock, 2, proto, 'bar', 'ms')
    eq_([6, 5], bar(5, 6))
    _timer_check(cl._sock, 3, proto, 'bar', 'ms')