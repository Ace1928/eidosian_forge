from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
@implementer(IProtocolNegotiationFactory)
class ServerNegotiationFactory(ServerFactory):
    """
    A L{ServerFactory} that has a set of acceptable protocols for NPN/ALPN
    negotiation.
    """

    def __init__(self, acceptableProtocols):
        """
        Create a L{ServerNegotiationFactory}.

        @param acceptableProtocols: The protocols the server will accept
            speaking after the TLS handshake is complete.
        @type acceptableProtocols: L{list} of L{bytes}
        """
        self._acceptableProtocols = acceptableProtocols

    def acceptableProtocols(self):
        """
        Returns a list of protocols that can be spoken by the connection
        factory in the form of ALPN tokens, as laid out in the IANA registry
        for ALPN tokens.

        @return: a list of ALPN tokens in order of preference.
        @rtype: L{list} of L{bytes}
        """
        return self._acceptableProtocols