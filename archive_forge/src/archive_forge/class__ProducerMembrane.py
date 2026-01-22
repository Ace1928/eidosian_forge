from __future__ import annotations
from typing import Callable, Iterable, Optional, cast
from zope.interface import directlyProvides, implementer, providedBy
from OpenSSL.SSL import Connection, Error, SysCallError, WantReadError, ZeroReturnError
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet._sslverify import _setAcceptableProtocols
from twisted.internet.interfaces import (
from twisted.internet.main import CONNECTION_LOST
from twisted.internet.protocol import Protocol
from twisted.protocols.policies import ProtocolWrapper, WrappingFactory
from twisted.python.failure import Failure
@implementer(IPushProducer)
class _ProducerMembrane:
    """
    Stand-in for producer registered with a L{TLSMemoryBIOProtocol} transport.

    Ensures that producer pause/resume events from the undelying transport are
    coordinated with pause/resume events from the TLS layer.

    @ivar _producer: The application-layer producer.
    """
    _producerPaused = False

    def __init__(self, producer):
        self._producer = producer

    def pauseProducing(self):
        """
        C{pauseProducing} the underlying producer, if it's not paused.
        """
        if self._producerPaused:
            return
        self._producerPaused = True
        self._producer.pauseProducing()

    def resumeProducing(self):
        """
        C{resumeProducing} the underlying producer, if it's paused.
        """
        if not self._producerPaused:
            return
        self._producerPaused = False
        self._producer.resumeProducing()

    def stopProducing(self):
        """
        C{stopProducing} the underlying producer.

        There is only a single source for this event, so it's simply passed
        on.
        """
        self._producer.stopProducing()