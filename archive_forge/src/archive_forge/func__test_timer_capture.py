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
def _test_timer_capture(cl, proto):
    with cl.timer('woo') as result:
        eq_(result.ms, None)
    assert isinstance(result.ms, float)