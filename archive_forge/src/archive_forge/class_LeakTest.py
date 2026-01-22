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
class LeakTest(AsyncTestCase):

    def tearDown(self):
        super().tearDown()
        gc.collect()

    def test_leaked_coroutine(self):
        event = Event()

        async def callback():
            try:
                await event.wait()
            except asyncio.CancelledError:
                pass
        self.io_loop.add_callback(callback)
        self.io_loop.add_callback(self.stop)
        self.wait()