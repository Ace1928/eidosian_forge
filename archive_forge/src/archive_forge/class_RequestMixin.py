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
class RequestMixin:

    def makeRequest(self, method=b'GET', clientAddress=None):
        """
        Create a L{DummyRequest} (change me to create a
        L{twisted.web.http.Request} instead).
        """
        if clientAddress is None:
            clientAddress = IPv4Address('TCP', 'localhost', 1234)
        request = DummyRequest(b'/')
        request.method = method
        request.client = clientAddress
        return request