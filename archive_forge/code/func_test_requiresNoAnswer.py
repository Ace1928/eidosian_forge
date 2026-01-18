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
def test_requiresNoAnswer(self):
    """
        Verify that a command that requires no answer is run.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    HELLO = b'world'
    c.callRemote(NoAnswerHello, hello=HELLO)
    p.flush()
    self.assertTrue(s.greeted)