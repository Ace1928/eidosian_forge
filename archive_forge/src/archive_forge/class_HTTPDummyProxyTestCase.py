import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
class HTTPDummyProxyTestCase(object):
    http_host = 'localhost'
    http_host_alt = '127.0.0.1'
    https_host = 'localhost'
    https_host_alt = '127.0.0.1'
    https_certs = DEFAULT_CERTS
    proxy_host = 'localhost'
    proxy_host_alt = '127.0.0.1'

    @classmethod
    def setup_class(cls):
        cls.io_loop = ioloop.IOLoop.current()
        app = web.Application([('.*', TestingApp)])
        cls.http_server, cls.http_port = run_tornado_app(app, cls.io_loop, None, 'http', cls.http_host)
        app = web.Application([('.*', TestingApp)])
        cls.https_server, cls.https_port = run_tornado_app(app, cls.io_loop, cls.https_certs, 'https', cls.http_host)
        app = web.Application([('.*', ProxyHandler)])
        cls.proxy_server, cls.proxy_port = run_tornado_app(app, cls.io_loop, None, 'http', cls.proxy_host)
        upstream_ca_certs = cls.https_certs.get('ca_certs', None)
        app = web.Application([('.*', ProxyHandler)], upstream_ca_certs=upstream_ca_certs)
        cls.https_proxy_server, cls.https_proxy_port = run_tornado_app(app, cls.io_loop, cls.https_certs, 'https', cls.proxy_host)
        cls.server_thread = run_loop_in_thread(cls.io_loop)

    @classmethod
    def teardown_class(cls):
        cls.io_loop.add_callback(cls.http_server.stop)
        cls.io_loop.add_callback(cls.https_server.stop)
        cls.io_loop.add_callback(cls.proxy_server.stop)
        cls.io_loop.add_callback(cls.https_proxy_server.stop)
        cls.io_loop.add_callback(cls.io_loop.stop)
        cls.server_thread.join()