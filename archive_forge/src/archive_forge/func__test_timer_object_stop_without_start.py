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
def _test_timer_object_stop_without_start(cl):
    with assert_raises(RuntimeError):
        cl.timer('foo').stop()