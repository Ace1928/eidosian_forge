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
def test_sync_gen_return(self):

    @gen.coroutine
    def f():
        raise gen.Return(42)
    result = (yield f())
    self.assertEqual(result, 42)
    self.finished = True