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
def test_quitBoxQuits(self):
    """
        Verify that commands with a responseType of QuitBox will in fact
        terminate the connection.
        """
    c, s, p = connectedServerAndClient(ServerClass=SimpleSymmetricCommandProtocol, ClientClass=SimpleSymmetricCommandProtocol)
    L = []
    HELLO = b'world'
    GOODBYE = b'everyone'
    c.sendHello(HELLO).addCallback(L.append)
    p.flush()
    self.assertEqual(L.pop()['hello'], HELLO)
    c.callRemote(Goodbye).addCallback(L.append)
    p.flush()
    self.assertEqual(L.pop()['goodbye'], GOODBYE)
    c.sendHello(HELLO).addErrback(L.append)
    L.pop().trap(error.ConnectionDone)