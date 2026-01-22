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
def _logoutTest(self):
    """
        Issue a request for an authentication-protected resource using valid
        credentials and then return the C{DummyRequest} instance which was
        used.

        This is a helper for tests about the behavior of the logout
        callback.
        """
    self.credentialFactories.append(BasicCredentialFactory('example.com'))

    class SlowerResource(Resource):

        def render(self, request):
            return NOT_DONE_YET
    self.avatar.putChild(self.childName, SlowerResource())
    request = self.makeRequest([self.childName])
    child = self._authorizedBasicLogin(request)
    request.render(child)
    self.assertEqual(self.realm.loggedOut, 0)
    return request