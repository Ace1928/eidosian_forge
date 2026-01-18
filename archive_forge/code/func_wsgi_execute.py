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
def wsgi_execute(self, environ=None):
    """
        Invoke the server's ``wsgi_application``.
        """
    self.wsgi_setup(environ)
    try:
        result = self.server.wsgi_application(self.wsgi_environ, self.wsgi_start_response)
        try:
            for chunk in result:
                self.wsgi_write_chunk(chunk)
            if not self.wsgi_headers_sent:
                self.wsgi_write_chunk(b'')
        finally:
            if hasattr(result, 'close'):
                result.close()
            result = None
    except socket.error as exce:
        self.wsgi_connection_drop(exce, environ)
        return
    except:
        if not self.wsgi_headers_sent:
            error_msg = 'Internal Server Error\n'
            self.wsgi_curr_headers = ('500 Internal Server Error', [('Content-type', 'text/plain'), ('Content-length', str(len(error_msg)))])
            self.wsgi_write_chunk(b'Internal Server Error\n')
        raise