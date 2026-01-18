import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def withRequest(f):
    """
    Decorator to cause the request to be passed as the first argument
    to the method.

    If an I{xmlrpc_} method is wrapped with C{withRequest}, the
    request object is passed as the first argument to that method.
    For example::

        @withRequest
        def xmlrpc_echo(self, request, s):
            return s

    @since: 10.2
    """
    f.withRequest = True
    return f