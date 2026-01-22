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
class Request(urllib.request.Request):
    """A custom Request object.

    urllib.request determines the request method heuristically (based on
    the presence or absence of data). We set the method
    statically.

    The Request object tracks:
    - the connection the request will be made on.
    - the authentication parameters needed to preventively set
      the authentication header once a first authentication have
       been made.
    """

    def __init__(self, method, url, data=None, headers={}, origin_req_host=None, unverifiable=False, connection=None, parent=None):
        urllib.request.Request.__init__(self, url, data, headers, origin_req_host, unverifiable)
        self.method = method
        self.connection = connection
        self.parent = parent
        self.redirected_to = None
        self.follow_redirections = False
        self.auth = {}
        self.proxy_auth = {}
        self.proxied_host = None

    def get_method(self):
        return self.method

    def set_proxy(self, proxy, type):
        """Set the proxy and remember the proxied host."""
        host, port = splitport(self.host)
        if port is None:
            if self.type == 'https':
                conn_class = HTTPSConnection
            else:
                conn_class = HTTPConnection
            port = conn_class.default_port
        self.proxied_host = '{}:{}'.format(host, port)
        urllib.request.Request.set_proxy(self, proxy, type)
        self.add_unredirected_header('Host', self.proxied_host)