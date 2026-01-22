import asyncio
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from collections.abc import Generator
import contextlib
import datetime
import functools
import socket
import subprocess
import sys
import threading
import time
import types
from unittest import mock
import unittest
from tornado.escape import native_str
from tornado import gen
from tornado.ioloop import IOLoop, TimeoutError, PeriodicCallback
from tornado.log import app_log
from tornado.testing import (
from tornado.test.util import (
from tornado.concurrent import Future
import typing
class TestIOLoop(AsyncTestCase):

    def test_add_callback_return_sequence(self):
        self.calls = 0
        loop = self.io_loop
        test = self
        old_add_callback = loop.add_callback

        def add_callback(self, callback, *args, **kwargs):
            test.calls += 1
            old_add_callback(callback, *args, **kwargs)
        loop.add_callback = types.MethodType(add_callback, loop)
        loop.add_callback(lambda: {})
        loop.add_callback(lambda: [])
        loop.add_timeout(datetime.timedelta(milliseconds=50), loop.stop)
        loop.start()
        self.assertLess(self.calls, 10)

    @skipOnTravis
    def test_add_callback_wakeup(self):

        def callback():
            self.called = True
            self.stop()

        def schedule_callback():
            self.called = False
            self.io_loop.add_callback(callback)
            self.start_time = time.time()
        self.io_loop.add_timeout(self.io_loop.time(), schedule_callback)
        self.wait()
        self.assertAlmostEqual(time.time(), self.start_time, places=2)
        self.assertTrue(self.called)

    @skipOnTravis
    def test_add_callback_wakeup_other_thread(self):

        def target():
            time.sleep(0.01)
            self.stop_time = time.time()
            self.io_loop.add_callback(self.stop)
        thread = threading.Thread(target=target)
        self.io_loop.add_callback(thread.start)
        self.wait()
        delta = time.time() - self.stop_time
        self.assertLess(delta, 0.1)
        thread.join()

    def test_add_timeout_timedelta(self):
        self.io_loop.add_timeout(datetime.timedelta(microseconds=1), self.stop)
        self.wait()

    def test_multiple_add(self):
        sock, port = bind_unused_port()
        try:
            self.io_loop.add_handler(sock.fileno(), lambda fd, events: None, IOLoop.READ)
            self.assertRaises(Exception, self.io_loop.add_handler, sock.fileno(), lambda fd, events: None, IOLoop.READ)
        finally:
            self.io_loop.remove_handler(sock.fileno())
            sock.close()

    def test_remove_without_add(self):
        sock, port = bind_unused_port()
        try:
            self.io_loop.remove_handler(sock.fileno())
        finally:
            sock.close()

    def test_add_callback_from_signal(self):
        with ignore_deprecation():
            self.io_loop.add_callback_from_signal(self.stop)
        self.wait()

    def test_add_callback_from_signal_other_thread(self):
        other_ioloop = IOLoop()
        thread = threading.Thread(target=other_ioloop.start)
        thread.start()
        with ignore_deprecation():
            other_ioloop.add_callback_from_signal(other_ioloop.stop)
        thread.join()
        other_ioloop.close()

    def test_add_callback_while_closing(self):
        closing = threading.Event()

        def target():
            other_ioloop.add_callback(other_ioloop.stop)
            other_ioloop.start()
            closing.set()
            other_ioloop.close(all_fds=True)
        other_ioloop = IOLoop()
        thread = threading.Thread(target=target)
        thread.start()
        closing.wait()
        for i in range(1000):
            other_ioloop.add_callback(lambda: None)

    @skipIfNonUnix
    def test_read_while_writeable(self):
        client, server = socket.socketpair()
        try:

            def handler(fd, events):
                self.assertEqual(events, IOLoop.READ)
                self.stop()
            self.io_loop.add_handler(client.fileno(), handler, IOLoop.READ)
            self.io_loop.add_timeout(self.io_loop.time() + 0.01, functools.partial(server.send, b'asdf'))
            self.wait()
            self.io_loop.remove_handler(client.fileno())
        finally:
            client.close()
            server.close()

    def test_remove_timeout_after_fire(self):
        handle = self.io_loop.add_timeout(self.io_loop.time(), self.stop)
        self.wait()
        self.io_loop.remove_timeout(handle)

    def test_remove_timeout_cleanup(self):
        for i in range(2000):
            timeout = self.io_loop.add_timeout(self.io_loop.time() + 3600, lambda: None)
            self.io_loop.remove_timeout(timeout)
        self.io_loop.add_callback(lambda: self.io_loop.add_callback(self.stop))
        self.wait()

    def test_remove_timeout_from_timeout(self):
        calls = [False, False]
        now = self.io_loop.time()

        def t1():
            calls[0] = True
            self.io_loop.remove_timeout(t2_handle)
        self.io_loop.add_timeout(now + 0.01, t1)

        def t2():
            calls[1] = True
        t2_handle = self.io_loop.add_timeout(now + 0.02, t2)
        self.io_loop.add_timeout(now + 0.03, self.stop)
        time.sleep(0.03)
        self.wait()
        self.assertEqual(calls, [True, False])

    def test_timeout_with_arguments(self):
        results = []
        self.io_loop.add_timeout(self.io_loop.time(), results.append, 1)
        self.io_loop.add_timeout(datetime.timedelta(seconds=0), results.append, 2)
        self.io_loop.call_at(self.io_loop.time(), results.append, 3)
        self.io_loop.call_later(0, results.append, 4)
        self.io_loop.call_later(0, self.stop)
        self.wait()
        self.assertEqual(sorted(results), [1, 2, 3, 4])

    def test_add_timeout_return(self):
        handle = self.io_loop.add_timeout(self.io_loop.time(), lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_at_return(self):
        handle = self.io_loop.call_at(self.io_loop.time(), lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_call_later_return(self):
        handle = self.io_loop.call_later(0, lambda: None)
        self.assertFalse(handle is None)
        self.io_loop.remove_timeout(handle)

    def test_close_file_object(self):
        """When a file object is used instead of a numeric file descriptor,
        the object should be closed (by IOLoop.close(all_fds=True),
        not just the fd.
        """

        class SocketWrapper(object):

            def __init__(self, sockobj):
                self.sockobj = sockobj
                self.closed = False

            def fileno(self):
                return self.sockobj.fileno()

            def close(self):
                self.closed = True
                self.sockobj.close()
        sockobj, port = bind_unused_port()
        socket_wrapper = SocketWrapper(sockobj)
        io_loop = IOLoop()
        io_loop.add_handler(socket_wrapper, lambda fd, events: None, IOLoop.READ)
        io_loop.close(all_fds=True)
        self.assertTrue(socket_wrapper.closed)

    def test_handler_callback_file_object(self):
        """The handler callback receives the same fd object it passed in."""
        server_sock, port = bind_unused_port()
        fds = []

        def handle_connection(fd, events):
            fds.append(fd)
            conn, addr = server_sock.accept()
            conn.close()
            self.stop()
        self.io_loop.add_handler(server_sock, handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(('127.0.0.1', port))
            self.wait()
        self.io_loop.remove_handler(server_sock)
        self.io_loop.add_handler(server_sock.fileno(), handle_connection, IOLoop.READ)
        with contextlib.closing(socket.socket()) as client_sock:
            client_sock.connect(('127.0.0.1', port))
            self.wait()
        self.assertIs(fds[0], server_sock)
        self.assertEqual(fds[1], server_sock.fileno())
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_mixed_fd_fileobj(self):
        server_sock, port = bind_unused_port()

        def f(fd, events):
            pass
        self.io_loop.add_handler(server_sock, f, IOLoop.READ)
        with self.assertRaises(Exception):
            self.io_loop.add_handler(server_sock.fileno(), f, IOLoop.READ)
        self.io_loop.remove_handler(server_sock.fileno())
        server_sock.close()

    def test_reentrant(self):
        """Calling start() twice should raise an error, not deadlock."""
        returned_from_start = [False]
        got_exception = [False]

        def callback():
            try:
                self.io_loop.start()
                returned_from_start[0] = True
            except Exception:
                got_exception[0] = True
            self.stop()
        self.io_loop.add_callback(callback)
        self.wait()
        self.assertTrue(got_exception[0])
        self.assertFalse(returned_from_start[0])

    def test_exception_logging(self):
        """Uncaught exceptions get logged by the IOLoop."""
        self.io_loop.add_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_exception_logging_future(self):
        """The IOLoop examines exceptions from Futures and logs them."""

        @gen.coroutine
        def callback():
            self.io_loop.add_callback(self.stop)
            1 / 0
        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_exception_logging_native_coro(self):
        """The IOLoop examines exceptions from awaitables and logs them."""

        async def callback():
            self.io_loop.add_callback(self.io_loop.add_callback, self.stop)
            1 / 0
        self.io_loop.add_callback(callback)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    def test_spawn_callback(self):
        self.io_loop.add_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()
        self.io_loop.spawn_callback(lambda: 1 / 0)
        self.io_loop.add_callback(self.stop)
        with ExpectLog(app_log, 'Exception in callback'):
            self.wait()

    @skipIfNonUnix
    def test_remove_handler_from_handler(self):
        client, server = socket.socketpair()
        try:
            client.send(b'abc')
            server.send(b'abc')
            chunks = []

            def handle_read(fd, events):
                chunks.append(fd.recv(1024))
                if fd is client:
                    self.io_loop.remove_handler(server)
                else:
                    self.io_loop.remove_handler(client)
            self.io_loop.add_handler(client, handle_read, self.io_loop.READ)
            self.io_loop.add_handler(server, handle_read, self.io_loop.READ)
            self.io_loop.call_later(0.1, self.stop)
            self.wait()
            self.assertEqual(chunks, [b'abc'])
        finally:
            client.close()
            server.close()

    @skipIfNonUnix
    @gen_test
    def test_init_close_race(self):

        def f():
            for i in range(10):
                loop = IOLoop(make_current=False)
                loop.close()
        yield gen.multi([self.io_loop.run_in_executor(None, f) for i in range(2)])

    def test_explicit_asyncio_loop(self):
        asyncio_loop = asyncio.new_event_loop()
        loop = IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        assert loop.asyncio_loop is asyncio_loop
        with self.assertRaises(RuntimeError):
            IOLoop(asyncio_loop=asyncio_loop, make_current=False)
        loop.close()