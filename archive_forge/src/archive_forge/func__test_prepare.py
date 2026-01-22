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
def _test_prepare(cl, proto):
    tests = (('foo:1|c', ('foo', '1|c', 1)), ('bar:50|ms|@0.5', ('bar', '50|ms', 0.5)), ('baz:23|g', ('baz', '23|g', 1)))

    def _check(o, s, v, r):
        with mock.patch.object(random, 'random', lambda: -1):
            eq_(o, cl._prepare(s, v, r))
    for o, (s, v, r) in tests:
        _check(o, s, v, r)