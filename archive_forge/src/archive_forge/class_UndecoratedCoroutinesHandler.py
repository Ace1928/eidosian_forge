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
class UndecoratedCoroutinesHandler(RequestHandler):

    @gen.coroutine
    def prepare(self):
        self.chunks = []
        yield gen.moment
        self.chunks.append('1')

    @gen.coroutine
    def get(self):
        self.chunks.append('2')
        yield gen.moment
        self.chunks.append('3')
        yield gen.moment
        self.write(''.join(self.chunks))