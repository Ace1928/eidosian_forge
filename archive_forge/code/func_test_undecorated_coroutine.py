from tornado import gen, ioloop
from tornado.httpserver import HTTPServer
from tornado.locks import Event
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, bind_unused_port, gen_test
from tornado.web import Application
import asyncio
import contextlib
import inspect
import gc
import os
import platform
import sys
import traceback
import unittest
import warnings
@unittest.skipIf(platform.python_implementation() == 'PyPy', 'pypy destructor warnings cannot be silenced')
@unittest.skipIf(sys.version_info >= (3, 12), 'py312 has its own check for test case returns')
def test_undecorated_coroutine(self):

    class Test(AsyncTestCase):

        async def test_coro(self):
            pass
    test = Test('test_coro')
    result = unittest.TestResult()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        test.run(result)
    self.assertEqual(len(result.errors), 1)
    self.assertIn('should be decorated', result.errors[0][1])