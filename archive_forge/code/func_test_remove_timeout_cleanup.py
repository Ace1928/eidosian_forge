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
def test_remove_timeout_cleanup(self):
    for i in range(2000):
        timeout = self.io_loop.add_timeout(self.io_loop.time() + 3600, lambda: None)
        self.io_loop.remove_timeout(timeout)
    self.io_loop.add_callback(lambda: self.io_loop.add_callback(self.stop))
    self.wait()