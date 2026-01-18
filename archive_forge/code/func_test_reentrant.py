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
def test_reentrant(self):
    """Calling start() twice should raise an error, not deadlock."""
    returned_from_start = [False]
    got_exception = [False]

    def callback():
        try:
            self.io_loop.start()
            returned_from_start[0] = True
        except Exception:
            got_exception[0] = True
        self.stop()
    self.io_loop.add_callback(callback)
    self.wait()
    self.assertTrue(got_exception[0])
    self.assertFalse(returned_from_start[0])