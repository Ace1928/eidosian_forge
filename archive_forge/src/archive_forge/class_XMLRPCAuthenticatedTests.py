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
class XMLRPCAuthenticatedTests(XMLRPCTests):
    """
    Test with authenticated proxy. We run this with the same input/output as
    above.
    """
    user = b'username'
    password = b'asecret'

    def setUp(self):
        self.p = reactor.listenTCP(0, server.Site(TestAuthHeader()), interface='127.0.0.1')
        self.port = self.p.getHost().port
        self.factories = []

    def test_authInfoInURL(self):
        url = 'http://%s:%s@127.0.0.1:%d/' % (nativeString(self.user), nativeString(self.password), self.port)
        p = xmlrpc.Proxy(networkString(url))
        d = p.callRemote('authinfo')
        d.addCallback(self.assertEqual, [self.user, self.password])
        return d

    def test_explicitAuthInfo(self):
        p = xmlrpc.Proxy(networkString('http://127.0.0.1:%d/' % (self.port,)), self.user, self.password)
        d = p.callRemote('authinfo')
        d.addCallback(self.assertEqual, [self.user, self.password])
        return d

    def test_longPassword(self):
        """
        C{QueryProtocol} uses the C{base64.b64encode} function to encode user
        name and password in the I{Authorization} header, so that it doesn't
        embed new lines when using long inputs.
        """
        longPassword = self.password * 40
        p = xmlrpc.Proxy(networkString('http://127.0.0.1:%d/' % (self.port,)), self.user, longPassword)
        d = p.callRemote('authinfo')
        d.addCallback(self.assertEqual, [self.user, longPassword])
        return d

    def test_explicitAuthInfoOverride(self):
        p = xmlrpc.Proxy(networkString('http://wrong:info@127.0.0.1:%d/' % (self.port,)), self.user, self.password)
        d = p.callRemote('authinfo')
        d.addCallback(self.assertEqual, [self.user, self.password])
        return d