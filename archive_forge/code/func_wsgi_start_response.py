import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def wsgi_start_response(self, status, response_headers, exc_info=None):
    if exc_info:
        try:
            if self.wsgi_headers_sent:
                raise exc_info
            else:
                pass
        finally:
            exc_info = None
    elif self.wsgi_curr_headers:
        assert 0, 'Attempt to set headers a second time w/o an exc_info'
    self.wsgi_curr_headers = (status, response_headers)
    return self.wsgi_write_chunk