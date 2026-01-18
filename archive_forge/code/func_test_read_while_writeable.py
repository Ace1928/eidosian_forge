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
@skipIfNonUnix
def test_read_while_writeable(self):
    client, server = socket.socketpair()
    try:

        def handler(fd, events):
            self.assertEqual(events, IOLoop.READ)
            self.stop()
        self.io_loop.add_handler(client.fileno(), handler, IOLoop.READ)
        self.io_loop.add_timeout(self.io_loop.time() + 0.01, functools.partial(server.send, b'asdf'))
        self.wait()
        self.io_loop.remove_handler(client.fileno())
    finally:
        client.close()
        server.close()