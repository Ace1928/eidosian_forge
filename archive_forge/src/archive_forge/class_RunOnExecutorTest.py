from concurrent import futures
import logging
import re
import socket
import typing
import unittest
from tornado.concurrent import (
from tornado.escape import utf8, to_unicode
from tornado import gen
from tornado.iostream import IOStream
from tornado.tcpserver import TCPServer
from tornado.testing import AsyncTestCase, bind_unused_port, gen_test
class RunOnExecutorTest(AsyncTestCase):

    @gen_test
    def test_no_calling(self):

        class Object(object):

            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor
            def f(self):
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_no_args(self):

        class Object(object):

            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_call_with_executor(self):

        class Object(object):

            def __init__(self):
                self.__executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor(executor='_Object__executor')
            def f(self):
                return 42
        o = Object()
        answer = (yield o.f())
        self.assertEqual(answer, 42)

    @gen_test
    def test_async_await(self):

        class Object(object):

            def __init__(self):
                self.executor = futures.thread.ThreadPoolExecutor(1)

            @run_on_executor()
            def f(self):
                return 42
        o = Object()

        async def f():
            answer = await o.f()
            return answer
        result = (yield f())
        self.assertEqual(result, 42)