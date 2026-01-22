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
class TestIOLoopConfiguration(unittest.TestCase):

    def run_python(self, *statements):
        stmt_list = ['from tornado.ioloop import IOLoop', 'classname = lambda x: x.__class__.__name__'] + list(statements)
        args = [sys.executable, '-c', '; '.join(stmt_list)]
        return native_str(subprocess.check_output(args)).strip()

    def test_default(self):
        cls = self.run_python('print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')
        cls = self.run_python('print(classname(IOLoop()))')
        self.assertEqual(cls, 'AsyncIOLoop')

    def test_asyncio(self):
        cls = self.run_python('IOLoop.configure("tornado.platform.asyncio.AsyncIOLoop")', 'print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')

    def test_asyncio_main(self):
        cls = self.run_python('from tornado.platform.asyncio import AsyncIOMainLoop', 'AsyncIOMainLoop().install()', 'print(classname(IOLoop.current()))')
        self.assertEqual(cls, 'AsyncIOMainLoop')