import asyncio
from concurrent import futures
import gc
import datetime
import platform
import sys
import time
import weakref
import unittest
from tornado.concurrent import Future
from tornado.log import app_log
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import skipOnTravis, skipNotCPython
from tornado.web import Application, RequestHandler, HTTPError
from tornado import gen
import typing
@gen_test
def test_py3_leak_exception_context(self):

    class LeakedException(Exception):
        pass

    @gen.coroutine
    def inner(iteration):
        raise LeakedException(iteration)
    try:
        yield inner(1)
    except LeakedException as e:
        self.assertEqual(str(e), '1')
        self.assertIsNone(e.__context__)
    try:
        yield inner(2)
    except LeakedException as e:
        self.assertEqual(str(e), '2')
        self.assertIsNone(e.__context__)
    self.finished = True