import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def addIntrospection(xmlrpc):
    """
    Add Introspection support to an XMLRPC server.

    @param xmlrpc: the XMLRPC server to add Introspection support to.
    @type xmlrpc: L{XMLRPC}
    """
    xmlrpc.putSubHandler('system', XMLRPCIntrospection(xmlrpc))