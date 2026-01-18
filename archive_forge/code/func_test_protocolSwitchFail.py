import datetime
import decimal
from typing import ClassVar, Dict, Type, TypeVar
from unittest import skipIf
from zope.interface import implementer
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import address, defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.protocols import amp
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.test import iosim
from twisted.trial.unittest import TestCase
def test_protocolSwitchFail(self, switcher=SimpleSymmetricCommandProtocol):
    """
        Verify that if we try to switch protocols and it fails, the connection
        stays up and we can go back to speaking AMP.
        """
    self.testSucceeded = False
    serverDeferred = defer.Deferred()
    serverProto = switcher(serverDeferred)
    clientDeferred = defer.Deferred()
    clientProto = switcher(clientDeferred)
    c, s, p = connectedServerAndClient(ServerClass=lambda: serverProto, ClientClass=lambda: clientProto)
    L = []
    c.switchToTestProtocol(fail=True).addErrback(L.append)
    p.flush()
    L.pop().trap(UnknownProtocol)
    self.assertFalse(self.testSucceeded)
    c.sendHello(b'world').addCallback(L.append)
    p.flush()
    self.assertEqual(L.pop()['hello'], b'world')