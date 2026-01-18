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
def wsgi_connection_drop(self, exce, environ=None):
    """
        Override this if you're interested in socket exceptions, such
        as when the user clicks 'Cancel' during a file download.
        """
    pass