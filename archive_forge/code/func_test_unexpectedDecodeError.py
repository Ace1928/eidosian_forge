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
def test_unexpectedDecodeError(self):
    """
        Any unexpected exception raised by the credential factory's C{decode}
        method results in a 500 response code and causes the exception to be
        logged.
        """
    logObserver = EventLoggingObserver.createWithCleanup(self, globalLogPublisher)

    class UnexpectedException(Exception):
        pass

    class BadFactory:
        scheme = b'bad'

        def getChallenge(self, client):
            return {}

        def decode(self, response, request):
            raise UnexpectedException()
    self.credentialFactories.append(BadFactory())
    request = self.makeRequest([self.childName])
    request.requestHeaders.addRawHeader(b'authorization', b'Bad abc')
    child = getChildForRequest(self.wrapper, request)
    request.render(child)
    self.assertEqual(request.responseCode, 500)
    self.assertEquals(1, len(logObserver))
    self.assertIsInstance(logObserver[0]['log_failure'].value, UnexpectedException)
    self.assertEqual(len(self.flushLoggedErrors(UnexpectedException)), 1)