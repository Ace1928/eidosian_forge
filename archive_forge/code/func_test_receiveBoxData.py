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
def test_receiveBoxData(self):
    """
        When a binary box protocol receives the serialized form of an AMP box,
        it should emit a similar box to its boxReceiver.
        """
    a = amp.BinaryBoxProtocol(self)
    a.dataReceived(amp.Box({b'testKey': b'valueTest', b'anotherKey': b'anotherValue'}).serialize())
    self.assertEqual(self.boxes, [amp.Box({b'testKey': b'valueTest', b'anotherKey': b'anotherValue'})])