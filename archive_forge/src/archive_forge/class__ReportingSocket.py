import base64
import cgi
import errno
import http.client
import os
import re
import socket
import ssl
import sys
import time
import urllib
import urllib.request
import weakref
from urllib.parse import urlencode, urljoin, urlparse
from ... import __version__ as breezy_version
from ... import config, debug, errors, osutils, trace, transport, ui, urlutils
from ...bzr.smart import medium
from ...trace import mutter, mutter_callsite
from ...transport import ConnectedTransport, NoSuchFile, UnusableRedirect
from . import default_user_agent, ssl
from .response import handle_response
class _ReportingSocket:

    def __init__(self, sock, report_activity=None):
        self.sock = sock
        self._report_activity = report_activity

    def report_activity(self, size, direction):
        if self._report_activity:
            self._report_activity(size, direction)

    def sendall(self, s, *args):
        self.sock.sendall(s, *args)
        self.report_activity(len(s), 'write')

    def recv(self, *args):
        s = self.sock.recv(*args)
        self.report_activity(len(s), 'read')
        return s

    def makefile(self, mode='r', bufsize=-1):
        fsock = self.sock.makefile(mode, 65536)
        return _ReportingFileSocket(fsock, self._report_activity)

    def __getattr__(self, name):
        return getattr(self.sock, name)