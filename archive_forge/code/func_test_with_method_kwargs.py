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
def test_with_method_kwargs(self):

    @gen_test
    def test_with_kwargs(self, **kwargs):
        self.assertDictEqual(kwargs, {'test': 'test'})
        yield gen.moment
    test_with_kwargs(self, test='test')
    self.finished = True