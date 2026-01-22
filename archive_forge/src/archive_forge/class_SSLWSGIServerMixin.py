import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
class SSLWSGIServerMixin:

    def finish_request(self, request, client_address):
        here = os.path.join(os.path.dirname(__file__), '..', 'tests')
        if not os.path.isdir(here):
            here = os.path.join(os.path.dirname(os.__file__), 'test', 'test_asyncio')
        keyfile = os.path.join(here, 'ssl_key.pem')
        certfile = os.path.join(here, 'ssl_cert.pem')
        ssock = ssl.wrap_socket(request, keyfile=keyfile, certfile=certfile, server_side=True)
        try:
            self.RequestHandlerClass(ssock, client_address, self)
            ssock.close()
        except OSError:
            pass