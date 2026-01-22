from urllib.parse import quote as urlquote, urlparse, urlunparse
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory
from twisted.web.http import _QUEUED_SENTINEL, HTTPChannel, HTTPClient, Request
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET
class ReverseProxy(HTTPChannel):
    """
    Implements a simple reverse proxy.

    For details of usage, see the file examples/reverse-proxy.py.
    """
    requestFactory = ReverseProxyRequest