import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server
def xmlrpc_listMethods(self):
    """
        Return a list of the method names implemented by this server.
        """
    functions = []
    todo = [(self._xmlrpc_parent, '')]
    while todo:
        obj, prefix = todo.pop(0)
        functions.extend([prefix + name for name in obj.listProcedures()])
        todo.extend([(obj.getSubHandler(name), prefix + name + obj.separator) for name in obj.getSubHandlerPrefixes()])
    return functions