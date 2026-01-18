from zope.interface import implementer
from twisted.internet import defer, interfaces, reactor
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IAddress, IPullProducer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def test_serverPeer(self):
    """
        Like C{test_serverHost} but for L{ITransport.getPeer}
        """
    return self._hostpeertest('getPeer', True)