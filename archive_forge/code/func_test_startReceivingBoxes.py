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
def test_startReceivingBoxes(self):
    """
        When L{amp.BinaryBoxProtocol} is connected to a transport, it calls
        C{startReceivingBoxes} on its L{IBoxReceiver} with itself as the
        L{IBoxSender} parameter.
        """
    protocol = amp.BinaryBoxProtocol(self)
    protocol.makeConnection(None)
    self.assertIs(self._boxSender, protocol)