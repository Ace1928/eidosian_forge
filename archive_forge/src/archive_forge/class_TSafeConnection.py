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
class TSafeConnection(tsafe.Connection):

    def settimeout(self, *args):
        self._lock.acquire()
        try:
            return self._ssl_conn.settimeout(*args)
        finally:
            self._lock.release()

    def gettimeout(self):
        self._lock.acquire()
        try:
            return self._ssl_conn.gettimeout()
        finally:
            self._lock.release()