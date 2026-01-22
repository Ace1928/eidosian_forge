import base64
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.cred import error, portal
from twisted.cred.checkers import (
from twisted.cred.credentials import IUsernamePassword
from twisted.internet.address import IPv4Address
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial import unittest
from twisted.web._auth import basic, digest
from twisted.web._auth.basic import BasicCredentialFactory
from twisted.web._auth.wrapper import HTTPAuthSessionWrapper, UnauthorizedResource
from twisted.web.iweb import ICredentialFactory
from twisted.web.resource import IResource, Resource, getChildForRequest
from twisted.web.server import NOT_DONE_YET
from twisted.web.static import Data
from twisted.web.test.test_web import DummyRequest
def _invalidAuthorizationTest(self, response):
    """
        Create a request with the given value as the value of an
        I{Authorization} header and perform resource traversal with it,
        starting at C{self.wrapper}.  Assert that the result is a 401 response
        code.  Return a L{Deferred} which fires when this is all done.
        """
    self.credentialFactories.append(BasicCredentialFactory('example.com'))
    request = self.makeRequest([self.childName])
    request.requestHeaders.addRawHeader(b'authorization', response)
    child = getChildForRequest(self.wrapper, request)
    d = request.notifyFinish()

    def cbFinished(result):
        self.assertEqual(request.responseCode, 401)
    d.addCallback(cbFinished)
    request.render(child)
    return d