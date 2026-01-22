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
class _ConnFixer(object):
    """ wraps a socket connection so it implements makefile """

    def __init__(self, conn):
        self.__conn = conn

    def makefile(self, mode, bufsize):
        return socket._fileobject(self.__conn, mode, bufsize)

    def __getattr__(self, attrib):
        return getattr(self.__conn, attrib)