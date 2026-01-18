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
@skipIf(sslSkip, 'OpenSSL not present')
def test_sslTimeout(self):
    """
        For I{HTTPS} URIs, L{xmlrpc.Proxy.callRemote} passes the value it
        received for the C{connectTimeout} parameter as the C{timeout} argument
        to the underlying connectSSL call.
        """
    reactor = MemoryReactor()
    proxy = xmlrpc.Proxy(b'https://127.0.0.1:69', connectTimeout=3.0, reactor=reactor)
    proxy.callRemote('someMethod')
    self.assertEqual(reactor.sslClients[0][4], 3.0)