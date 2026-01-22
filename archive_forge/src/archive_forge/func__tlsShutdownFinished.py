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
def _tlsShutdownFinished(self, reason):
    """
        Called when TLS connection has gone away; tell underlying transport to
        disconnect.

        @param reason: a L{Failure} whose value is an L{Exception} if we want to
            report that failure through to the wrapped protocol's
            C{connectionLost}, or L{None} if the C{reason} that
            C{connectionLost} should receive should be coming from the
            underlying transport.
        @type reason: L{Failure} or L{None}
        """
    if reason is not None:
        if _representsEOF(reason.value):
            reason = Failure(CONNECTION_LOST)
    if self._reason is None:
        self._reason = reason
    self._lostTLSConnection = True
    self._flushSendBIO()
    self.transport.loseConnection()