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
def test_protocolSwitchLoseConnection(self):
    """
        When the protocol is switched, it should notify its nested protocol of
        disconnection.
        """

    class Loser(protocol.Protocol):
        reason = None

        def connectionLost(self, reason):
            self.reason = reason
    connectionLoser = Loser()
    a = amp.BinaryBoxProtocol(self)
    a.makeConnection(self)
    a._lockForSwitch()
    a._switchTo(connectionLoser)
    connectionFailure = Failure(RuntimeError())
    a.connectionLost(connectionFailure)
    self.assertEqual(connectionLoser.reason, connectionFailure)