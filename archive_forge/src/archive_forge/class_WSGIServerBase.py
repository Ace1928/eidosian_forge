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
class WSGIServerBase(SecureHTTPServer):

    def __init__(self, wsgi_application, server_address, RequestHandlerClass=None, ssl_context=None, request_queue_size=None):
        SecureHTTPServer.__init__(self, server_address, RequestHandlerClass, ssl_context, request_queue_size=request_queue_size)
        self.wsgi_application = wsgi_application
        self.wsgi_socket_timeout = None

    def get_request(self):
        conn, info = SecureHTTPServer.get_request(self)
        if self.wsgi_socket_timeout:
            conn.settimeout(self.wsgi_socket_timeout)
        return (conn, info)