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
def test_protocolSwitchLoseClientConnection(self):
    """
        When the protocol is switched, it should notify its nested client
        protocol factory of disconnection.
        """

    class ClientLoser:
        reason = None

        def clientConnectionLost(self, connector, reason):
            self.reason = reason
    a = amp.BinaryBoxProtocol(self)
    connectionLoser = protocol.Protocol()
    clientLoser = ClientLoser()
    a.makeConnection(self)
    a._lockForSwitch()
    a._switchTo(connectionLoser, clientLoser)
    connectionFailure = Failure(RuntimeError())
    a.connectionLost(connectionFailure)
    self.assertEqual(clientLoser.reason, connectionFailure)