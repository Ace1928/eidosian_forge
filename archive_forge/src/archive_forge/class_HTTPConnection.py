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
class HTTPConnection(AbstractHTTPConnection, http.client.HTTPConnection):

    def __init__(self, host, port=None, proxied_host=None, report_activity=None, ca_certs=None):
        AbstractHTTPConnection.__init__(self, report_activity=report_activity)
        http.client.HTTPConnection.__init__(self, host, port)
        self.proxied_host = proxied_host

    def connect(self):
        if 'http' in debug.debug_flags:
            self._mutter_connect()
        http.client.HTTPConnection.connect(self)
        self._wrap_socket_for_reporting(self.sock)