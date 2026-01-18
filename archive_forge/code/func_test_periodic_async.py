import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
def test_periodic_async(self) -> None:
    counts = [0, 0]

    async def callback() -> None:
        counts[0] += 1
        await gen.sleep(0.025)
        counts[1] += 1
        if counts[1] == 3:
            pc.stop()
            self.io_loop.add_callback(self.stop)
    pc = PeriodicCallback(callback, 10)
    pc.start()
    self.wait()
    self.assertEqual(counts[0], 3)
    self.assertEqual(counts[1], 3)