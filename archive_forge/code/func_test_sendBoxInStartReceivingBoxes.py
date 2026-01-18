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
def test_sendBoxInStartReceivingBoxes(self):
    """
        The L{IBoxReceiver} which is started when L{amp.BinaryBoxProtocol} is
        connected to a transport can call C{sendBox} on the L{IBoxSender}
        passed to it before C{startReceivingBoxes} returns and have that box
        sent.
        """

    class SynchronouslySendingReceiver:

        def startReceivingBoxes(self, sender):
            sender.sendBox(amp.Box({b'foo': b'bar'}))
    transport = StringTransport()
    protocol = amp.BinaryBoxProtocol(SynchronouslySendingReceiver())
    protocol.makeConnection(transport)
    self.assertEqual(transport.value(), b'\x00\x03foo\x00\x03bar\x00\x00')