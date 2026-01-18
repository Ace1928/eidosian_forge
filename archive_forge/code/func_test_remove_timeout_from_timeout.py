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
def test_remove_timeout_from_timeout(self):
    calls = [False, False]
    now = self.io_loop.time()

    def t1():
        calls[0] = True
        self.io_loop.remove_timeout(t2_handle)
    self.io_loop.add_timeout(now + 0.01, t1)

    def t2():
        calls[1] = True
    t2_handle = self.io_loop.add_timeout(now + 0.02, t2)
    self.io_loop.add_timeout(now + 0.03, self.stop)
    time.sleep(0.03)
    self.wait()
    self.assertEqual(calls, [True, False])