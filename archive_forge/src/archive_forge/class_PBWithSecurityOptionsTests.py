import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
class PBWithSecurityOptionsTests(unittest.TestCase):
    """
    Test security customization.
    """

    def test_clientDefaultSecurityOptions(self):
        """
        By default, client broker should use C{jelly.globalSecurity} as
        security settings.
        """
        factory = pb.PBClientFactory()
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, jelly.globalSecurity)

    def test_serverDefaultSecurityOptions(self):
        """
        By default, server broker should use C{jelly.globalSecurity} as
        security settings.
        """
        factory = pb.PBServerFactory(Echoer())
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, jelly.globalSecurity)

    def test_clientSecurityCustomization(self):
        """
        Check that the security settings are passed from the client factory to
        the broker object.
        """
        security = jelly.SecurityOptions()
        factory = pb.PBClientFactory(security=security)
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, security)

    def test_serverSecurityCustomization(self):
        """
        Check that the security settings are passed from the server factory to
        the broker object.
        """
        security = jelly.SecurityOptions()
        factory = pb.PBServerFactory(Echoer(), security=security)
        broker = factory.buildProtocol(None)
        self.assertIs(broker.security, security)