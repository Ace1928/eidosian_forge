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
def test_requiresNoAnswerFail(self):
    """
        Verify that commands sent after a failed no-answer request do not complete.
        """
    L = []
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    HELLO = b'fuck you'
    c.callRemote(NoAnswerHello, hello=HELLO)
    p.flush()
    self.assertTrue(self.flushLoggedErrors(amp.RemoteAmpError))
    HELLO = b'world'
    c.callRemote(Hello, hello=HELLO).addErrback(L.append)
    p.flush()
    L.pop().trap(error.ConnectionDone)
    self.assertFalse(s.greeted)