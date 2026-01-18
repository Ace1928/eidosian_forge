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
def test_receiveBoxStateMachine(self):
    """
        When a binary box protocol receives:
            * a key
            * a value
            * an empty string
        it should emit a box and send it to its boxReceiver.
        """
    a = amp.BinaryBoxProtocol(self)
    a.stringReceived(b'hello')
    a.stringReceived(b'world')
    a.stringReceived(b'')
    self.assertEqual(self.boxes, [amp.AmpBox(hello=b'world')])