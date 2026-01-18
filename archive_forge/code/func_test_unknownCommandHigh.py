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
def test_unknownCommandHigh(self):
    """
        Verify that unknown commands using high-level APIs will be rejected with an
        error, but will NOT terminate the connection.
        """
    c, s, p = connectedServerAndClient()
    L = []

    def clearAndAdd(e):
        """
            You can't propagate the error...
            """
        e.trap(amp.UnhandledCommand)
        return 'OK'
    c.callRemote(WTF).addErrback(clearAndAdd).addCallback(L.append)
    p.flush()
    self.assertEqual(L.pop(), 'OK')
    HELLO = b'world'
    c.sendHello(HELLO).addCallback(L.append)
    p.flush()
    self.assertEqual(L[0][b'hello'], HELLO)