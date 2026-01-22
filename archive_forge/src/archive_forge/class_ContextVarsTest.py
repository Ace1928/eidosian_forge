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
@unittest.skipIf(contextvars is None, 'contextvars module not present')
class ContextVarsTest(AsyncTestCase):

    async def native_root(self, x):
        ctx_var.set(x)
        await self.inner(x)

    @gen.coroutine
    def gen_root(self, x):
        ctx_var.set(x)
        yield
        yield self.inner(x)

    async def inner(self, x):
        self.assertEqual(ctx_var.get(), x)
        await self.gen_inner(x)
        self.assertEqual(ctx_var.get(), x)
        ctx = contextvars.copy_context()
        await self.io_loop.run_in_executor(None, lambda: ctx.run(self.thread_inner, x))
        self.assertEqual(ctx_var.get(), x)
        await asyncio.get_event_loop().run_in_executor(None, lambda: ctx.run(self.thread_inner, x))
        self.assertEqual(ctx_var.get(), x)

    @gen.coroutine
    def gen_inner(self, x):
        self.assertEqual(ctx_var.get(), x)
        yield
        self.assertEqual(ctx_var.get(), x)

    def thread_inner(self, x):
        self.assertEqual(ctx_var.get(), x)

    @gen_test
    def test_propagate(self):
        yield [self.native_root(1), self.native_root(2), self.gen_root(3), self.gen_root(4)]

    @gen_test
    def test_reset(self):
        token = ctx_var.set(1)
        yield
        ctx_var.reset(token)

    @gen_test
    def test_propagate_to_first_yield_with_native_async_function(self):
        x = 10

        async def native_async_function():
            self.assertEqual(ctx_var.get(), x)
        ctx_var.set(x)
        yield native_async_function()