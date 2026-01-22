import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
class SocketDummyServerTestCase(object):
    """
    A simple socket-based server is created for this class that is good for
    exactly one request.
    """
    scheme = 'http'
    host = 'localhost'

    @classmethod
    def _start_server(cls, socket_handler):
        ready_event = threading.Event()
        cls.server_thread = SocketServerThread(socket_handler=socket_handler, ready_event=ready_event, host=cls.host)
        cls.server_thread.start()
        ready_event.wait(5)
        if not ready_event.is_set():
            raise Exception('most likely failed to start server')
        cls.port = cls.server_thread.port

    @classmethod
    def start_response_handler(cls, response, num=1, block_send=None):
        ready_event = threading.Event()

        def socket_handler(listener):
            for _ in range(num):
                ready_event.set()
                sock = listener.accept()[0]
                consume_socket(sock)
                if block_send:
                    block_send.wait()
                    block_send.clear()
                sock.send(response)
                sock.close()
        cls._start_server(socket_handler)
        return ready_event

    @classmethod
    def start_basic_handler(cls, **kw):
        return cls.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n', **kw)

    @classmethod
    def teardown_class(cls):
        if hasattr(cls, 'server_thread'):
            cls.server_thread.join(0.1)

    def assert_header_received(self, received_headers, header_name, expected_value=None):
        header_name = header_name.encode('ascii')
        if expected_value is not None:
            expected_value = expected_value.encode('ascii')
        header_titles = []
        for header in received_headers:
            key, value = header.split(b': ')
            header_titles.append(key)
            if key == header_name and expected_value is not None:
                assert value == expected_value
        assert header_name in header_titles