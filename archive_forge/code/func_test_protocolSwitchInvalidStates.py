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
def test_protocolSwitchInvalidStates(self):
    """
        In order to make sure the protocol never gets any invalid data sent
        into the middle of a box, it must be locked for switching before it is
        switched.  It can only be unlocked if the switch failed, and attempting
        to send a box while it is locked should raise an exception.
        """
    a = amp.BinaryBoxProtocol(self)
    a.makeConnection(self)
    sampleBox = amp.Box({b'some': b'data'})
    a._lockForSwitch()
    self.assertRaises(amp.ProtocolSwitched, a.sendBox, sampleBox)
    a._unlockFromSwitch()
    a.sendBox(sampleBox)
    self.assertEqual(b''.join(self.data), sampleBox.serialize())
    a._lockForSwitch()
    otherProto = TestProto(None, b'outgoing data')
    a._switchTo(otherProto)
    self.assertRaises(amp.ProtocolSwitched, a._unlockFromSwitch)