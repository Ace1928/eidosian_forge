import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def xmlrpc_methodHelp(self, method):
    """
        Return a documentation string describing the use of the given method.
        """
    method = self._xmlrpc_parent.lookupProcedure(method)
    return getattr(method, 'help', None) or getattr(method, '__doc__', None) or ''