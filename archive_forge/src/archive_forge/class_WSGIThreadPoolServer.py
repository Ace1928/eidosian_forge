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
class WSGIThreadPoolServer(ThreadPoolMixIn, WSGIServerBase):

    def __init__(self, wsgi_application, server_address, RequestHandlerClass=None, ssl_context=None, nworkers=10, daemon_threads=False, threadpool_options=None, request_queue_size=None):
        WSGIServerBase.__init__(self, wsgi_application, server_address, RequestHandlerClass, ssl_context, request_queue_size=request_queue_size)
        if threadpool_options is None:
            threadpool_options = {}
        ThreadPoolMixIn.__init__(self, nworkers, daemon_threads, **threadpool_options)