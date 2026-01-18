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
def test_with_method_args(self):

    @gen_test
    def test_with_args(self, *args):
        self.assertEqual(args, ('test',))
        yield gen.moment
    test_with_args(self, 'test')
    self.finished = True