import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def test_tcpTimeout(self):
    """
        For I{HTTP} URIs, L{xmlrpc.Proxy.callRemote} passes the value it
        received for the C{connectTimeout} parameter as the C{timeout} argument
        to the underlying connectTCP call.
        """
    reactor = MemoryReactor()
    proxy = xmlrpc.Proxy(b'http://127.0.0.1:69', connectTimeout=2.0, reactor=reactor)
    proxy.callRemote('someMethod')
    self.assertEqual(reactor.tcpClients[0][3], 2.0)