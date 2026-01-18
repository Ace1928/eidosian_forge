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
def test_undecorated_generator(self):

    class Test(AsyncTestCase):

        def test_gen(self):
            yield
    test = Test('test_gen')
    result = unittest.TestResult()
    test.run(result)
    self.assertEqual(len(result.errors), 1)
    self.assertIn('should be decorated', result.errors[0][1])