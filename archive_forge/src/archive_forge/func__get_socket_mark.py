import threading
from contextlib import contextmanager
import pytest
from tornado import ioloop, web
from dummyserver.handlers import TestingApp
from dummyserver.proxy import ProxyHandler
from dummyserver.server import (
from urllib3.connection import HTTPConnection
@classmethod
def _get_socket_mark(cls, sock, server):
    if server:
        port = sock.getpeername()[1]
    else:
        port = sock.getsockname()[1]
    return cls.MARK_FORMAT % (port,)