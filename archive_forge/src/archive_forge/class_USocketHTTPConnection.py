import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper
class USocketHTTPConnection(HTTPConnection):
    """
    HTTPConnection over a unix socket.
    """

    def __init__(self, path):
        HTTPConnection.__init__(self, 'localhost')
        self.path = path

    def __call__(self, *args, **kwargs):
        """
        Catch-all method just to present itself as a constructor for the
        HTTPConnection.
        """
        return self

    def connect(self):
        """
        Override the connect method and assign a unix socket as a transport.
        """
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(self.path)
        self.sock = sock
        atexit.register(lambda: os.remove(self.path))