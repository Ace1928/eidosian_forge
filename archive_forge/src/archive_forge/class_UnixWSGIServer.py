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
class UnixWSGIServer(UnixHTTPServer, WSGIServer):
    request_timeout = 2

    def server_bind(self):
        UnixHTTPServer.server_bind(self)
        self.setup_environ()

    def get_request(self):
        request, client_addr = super().get_request()
        request.settimeout(self.request_timeout)
        return (request, ('127.0.0.1', ''))