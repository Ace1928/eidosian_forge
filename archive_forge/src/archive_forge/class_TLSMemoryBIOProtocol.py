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
@implementer(ISystemHandle, INegotiated, ITransport)
class TLSMemoryBIOProtocol(ProtocolWrapper):
    """
    L{TLSMemoryBIOProtocol} is a protocol wrapper which uses OpenSSL via a
    memory BIO to encrypt bytes written to it before sending them on to the
    underlying transport and decrypts bytes received from the underlying
    transport before delivering them to the wrapped protocol.

    In addition to producer events from the underlying transport, the need to
    wait for reads before a write can proceed means the L{TLSMemoryBIOProtocol}
    may also want to pause a producer.  Pause/resume events are therefore
    merged using the L{_ProducerMembrane} wrapper.  Non-streaming (pull)
    producers are supported by wrapping them with L{_PullToPush}.

    Because TLS may need to wait for reads before writing, some writes may be
    buffered until a read occurs.

    @ivar _tlsConnection: The L{OpenSSL.SSL.Connection} instance which is
        encrypted and decrypting this connection.

    @ivar _lostTLSConnection: A flag indicating whether connection loss has
        already been dealt with (C{True}) or not (C{False}).  TLS disconnection
        is distinct from the underlying connection being lost.

    @ivar _appSendBuffer: application-level (cleartext) data that is waiting to
        be transferred to the TLS buffer, but can't be because the TLS
        connection is handshaking.
    @type _appSendBuffer: L{list} of L{bytes}

    @ivar _connectWrapped: A flag indicating whether or not to call
        C{makeConnection} on the wrapped protocol.  This is for the reactor's
        L{twisted.internet.interfaces.ITLSTransport.startTLS} implementation,
        since it has a protocol which it has already called C{makeConnection}
        on, and which has no interest in a new transport.  See #3821.

    @ivar _handshakeDone: A flag indicating whether or not the handshake is
        known to have completed successfully (C{True}) or not (C{False}).  This
        is used to control error reporting behavior.  If the handshake has not
        completed, the underlying L{OpenSSL.SSL.Error} will be passed to the
        application's C{connectionLost} method.  If it has completed, any
        unexpected L{OpenSSL.SSL.Error} will be turned into a
        L{ConnectionLost}.  This is weird; however, it is simply an attempt at
        a faithful re-implementation of the behavior provided by
        L{twisted.internet.ssl}.

    @ivar _reason: If an unexpected L{OpenSSL.SSL.Error} occurs which causes
        the connection to be lost, it is saved here.  If appropriate, this may
        be used as the reason passed to the application protocol's
        C{connectionLost} method.

    @ivar _producer: The current producer registered via C{registerProducer},
        or L{None} if no producer has been registered or a previous one was
        unregistered.

    @ivar _aborted: C{abortConnection} has been called.  No further data will
        be received to the wrapped protocol's C{dataReceived}.
    @type _aborted: L{bool}
    """
    _reason = None
    _handshakeDone = False
    _lostTLSConnection = False
    _producer = None
    _aborted = False

    def __init__(self, factory, wrappedProtocol, _connectWrapped=True):
        ProtocolWrapper.__init__(self, factory, wrappedProtocol)
        self._connectWrapped = _connectWrapped

    def getHandle(self):
        """
        Return the L{OpenSSL.SSL.Connection} object being used to encrypt and
        decrypt this connection.

        This is done for the benefit of L{twisted.internet.ssl.Certificate}'s
        C{peerFromTransport} and C{hostFromTransport} methods only.  A
        different system handle may be returned by future versions of this
        method.
        """
        return self._tlsConnection

    def makeConnection(self, transport):
        """
        Connect this wrapper to the given transport and initialize the
        necessary L{OpenSSL.SSL.Connection} with a memory BIO.
        """
        self._tlsConnection = self.factory._createConnection(self)
        self._appSendBuffer = []
        for interface in providedBy(transport):
            directlyProvides(self, interface)
        Protocol.makeConnection(self, transport)
        self.factory.registerProtocol(self)
        if self._connectWrapped:
            ProtocolWrapper.makeConnection(self, transport)
        self._checkHandshakeStatus()

    def _checkHandshakeStatus(self):
        """
        Ask OpenSSL to proceed with a handshake in progress.

        Initially, this just sends the ClientHello; after some bytes have been
        stuffed in to the C{Connection} object by C{dataReceived}, it will then
        respond to any C{Certificate} or C{KeyExchange} messages.
        """
        if self._aborted:
            return
        try:
            self._tlsConnection.do_handshake()
        except WantReadError:
            self._flushSendBIO()
        except Error:
            self._tlsShutdownFinished(Failure())
        else:
            self._handshakeDone = True
            if IHandshakeListener.providedBy(self.wrappedProtocol):
                self.wrappedProtocol.handshakeCompleted()

    def _flushSendBIO(self):
        """
        Read any bytes out of the send BIO and write them to the underlying
        transport.
        """
        try:
            bytes = self._tlsConnection.bio_read(2 ** 15)
        except WantReadError:
            pass
        else:
            self.transport.write(bytes)

    def _flushReceiveBIO(self):
        """
        Try to receive any application-level bytes which are now available
        because of a previous write into the receive BIO.  This will take
        care of delivering any application-level bytes which are received to
        the protocol, as well as handling of the various exceptions which
        can come from trying to get such bytes.
        """
        while not self._lostTLSConnection:
            try:
                bytes = self._tlsConnection.recv(2 ** 15)
            except WantReadError:
                break
            except ZeroReturnError:
                self._shutdownTLS()
                self._tlsShutdownFinished(None)
            except Error:
                failure = Failure()
                self._tlsShutdownFinished(failure)
            else:
                if not self._aborted:
                    ProtocolWrapper.dataReceived(self, bytes)
        self._flushSendBIO()

    def dataReceived(self, bytes):
        """
        Deliver any received bytes to the receive BIO and then read and deliver
        to the application any application-level data which becomes available
        as a result of this.
        """
        self._tlsConnection.bio_write(bytes)
        if not self._handshakeDone:
            self._checkHandshakeStatus()
            if not self._handshakeDone:
                return
        if self._appSendBuffer:
            self._unbufferPendingWrites()
        self._flushReceiveBIO()

    def _shutdownTLS(self):
        """
        Initiate, or reply to, the shutdown handshake of the TLS layer.
        """
        try:
            shutdownSuccess = self._tlsConnection.shutdown()
        except Error:
            shutdownSuccess = False
        self._flushSendBIO()
        if shutdownSuccess:
            self.transport.loseConnection()

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

    def connectionLost(self, reason):
        """
        Handle the possible repetition of calls to this method (due to either
        the underlying transport going away or due to an error at the TLS
        layer) and make sure the base implementation only gets invoked once.
        """
        if not self._lostTLSConnection:
            self._tlsConnection.bio_shutdown()
            self._flushReceiveBIO()
            self._lostTLSConnection = True
        reason = self._reason or reason
        self._reason = None
        self.connected = False
        ProtocolWrapper.connectionLost(self, reason)
        self._tlsConnection = None

    def loseConnection(self):
        """
        Send a TLS close alert and close the underlying connection.
        """
        if self.disconnecting or not self.connected:
            return
        if not self._handshakeDone and (not self._appSendBuffer):
            self.abortConnection()
        self.disconnecting = True
        if not self._appSendBuffer and self._producer is None:
            self._shutdownTLS()

    def abortConnection(self):
        """
        Tear down TLS state so that if the connection is aborted mid-handshake
        we don't deliver any further data from the application.
        """
        self._aborted = True
        self.disconnecting = True
        self._shutdownTLS()
        self.transport.abortConnection()

    def failVerification(self, reason):
        """
        Abort the connection during connection setup, giving a reason that
        certificate verification failed.

        @param reason: The reason that the verification failed; reported to the
            application protocol's C{connectionLost} method.
        @type reason: L{Failure}
        """
        self._reason = reason
        self.abortConnection()

    def write(self, bytes):
        """
        Process the given application bytes and send any resulting TLS traffic
        which arrives in the send BIO.

        If C{loseConnection} was called, subsequent calls to C{write} will
        drop the bytes on the floor.
        """
        if self.disconnecting and self._producer is None:
            return
        self._write(bytes)

    def _bufferedWrite(self, octets):
        """
        Put the given octets into L{TLSMemoryBIOProtocol._appSendBuffer}, and
        tell any listening producer that it should pause because we are now
        buffering.
        """
        self._appSendBuffer.append(octets)
        if self._producer is not None:
            self._producer.pauseProducing()

    def _unbufferPendingWrites(self):
        """
        Un-buffer all waiting writes in L{TLSMemoryBIOProtocol._appSendBuffer}.
        """
        pendingWrites, self._appSendBuffer = (self._appSendBuffer, [])
        for eachWrite in pendingWrites:
            self._write(eachWrite)
        if self._appSendBuffer:
            return
        if self._producer is not None:
            self._producer.resumeProducing()
            return
        if self.disconnecting:
            self._shutdownTLS()

    def _write(self, bytes):
        """
        Process the given application bytes and send any resulting TLS traffic
        which arrives in the send BIO.

        This may be called by C{dataReceived} with bytes that were buffered
        before C{loseConnection} was called, which is why this function
        doesn't check for disconnection but accepts the bytes regardless.
        """
        if self._lostTLSConnection:
            return
        bufferSize = 2 ** 14
        alreadySent = 0
        while alreadySent < len(bytes):
            toSend = bytes[alreadySent:alreadySent + bufferSize]
            try:
                sent = self._tlsConnection.send(toSend)
            except WantReadError:
                self._bufferedWrite(bytes[alreadySent:])
                break
            except Error:
                self._tlsShutdownFinished(Failure())
                break
            else:
                alreadySent += sent
                self._flushSendBIO()

    def writeSequence(self, iovec):
        """
        Write a sequence of application bytes by joining them into one string
        and passing them to L{write}.
        """
        self.write(b''.join(iovec))

    def getPeerCertificate(self):
        return self._tlsConnection.get_peer_certificate()

    @property
    def negotiatedProtocol(self):
        """
        @see: L{INegotiated.negotiatedProtocol}
        """
        protocolName = None
        try:
            protocolName = self._tlsConnection.get_alpn_proto_negotiated()
        except (NotImplementedError, AttributeError):
            pass
        if protocolName not in (b'', None):
            return protocolName
        try:
            protocolName = self._tlsConnection.get_next_proto_negotiated()
        except (NotImplementedError, AttributeError):
            pass
        if protocolName != b'':
            return protocolName
        return None

    def registerProducer(self, producer, streaming):
        if self._lostTLSConnection:
            producer.stopProducing()
            return
        if not streaming:
            producer = streamingProducer = _PullToPush(producer, self)
        producer = _ProducerMembrane(producer)
        self.transport.registerProducer(producer, True)
        self._producer = producer
        if not streaming:
            streamingProducer.startStreaming()

    def unregisterProducer(self):
        if self._producer is None:
            return
        if isinstance(self._producer._producer, _PullToPush):
            self._producer._producer.stopStreaming()
        self._producer = None
        self._producerPaused = False
        self.transport.unregisterProducer()
        if self.disconnecting and (not self._appSendBuffer):
            self._shutdownTLS()