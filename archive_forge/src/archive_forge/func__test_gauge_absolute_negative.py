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
def _test_gauge_absolute_negative(cl, proto):
    cl.gauge('foo', -5, delta=False)
    _sock_check(cl._sock, 1, 'foo:0|g\nfoo:-5|g')