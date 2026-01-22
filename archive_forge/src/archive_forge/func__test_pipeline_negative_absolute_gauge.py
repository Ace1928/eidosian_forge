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
def _test_pipeline_negative_absolute_gauge(cl, proto):
    with cl.pipeline() as pipe:
        pipe.gauge('foo', -10, delta=False)
        pipe.incr('bar')
    _sock_check(cl._sock, 1, proto, 'foo:0|g\nfoo:-10|g\nbar:1|c')