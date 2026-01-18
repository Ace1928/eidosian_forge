import base64
import xmlrpc.client as xmlrpclib
from urllib.parse import urlparse
from xmlrpc.client import Binary, Boolean, DateTime, Fault
from twisted.internet import defer, error, protocol
from twisted.logger import Logger
from twisted.python import failure, reflect
from twisted.python.compat import nativeString
from twisted.web import http, resource, server

        Call remote XML-RPC C{method} with given arguments.

        @return: a L{defer.Deferred} that will fire with the method response,
            or a failure if the method failed. Generally, the failure type will
            be L{Fault}, but you can also have an C{IndexError} on some buggy
            servers giving empty responses.

            If the deferred is cancelled before the request completes, the
            connection is closed and the deferred will fire with a
            L{defer.CancelledError}.
        