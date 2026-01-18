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
def test_selectParseResponse(self):
    """
        L{HTTPAuthSessionWrapper._selectParseHeader} returns a two-tuple giving
        the L{ICredentialFactory} to use to parse the header and a string
        containing the portion of the header which remains to be parsed.
        """
    basicAuthorization = b'Basic abcdef123456'
    self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (None, None))
    factory = BasicCredentialFactory('example.com')
    self.credentialFactories.append(factory)
    self.assertEqual(self.wrapper._selectParseHeader(basicAuthorization), (factory, b'abcdef123456'))