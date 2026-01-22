import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import (
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
class TunnelTestsMixin:
    """
    A mixin defining tests for L{TuntapPort}.

    These tests run against L{MemoryIOSystem} (proven equivalent to the real
    thing by the tests above) to avoid performing any real I/O.
    """

    def setUp(self):
        """
        Create an in-memory I/O system and set up a L{TuntapPort} against it.
        """
        self.name = b'tun0'
        self.system = MemoryIOSystem()
        self.system.registerSpecialDevice(Tunnel._DEVICE_NAME, Tunnel)
        self.protocol = self.factory.buildProtocol(TunnelAddress(self.helper.TUNNEL_TYPE, self.name))
        self.reactor = FSSetClock()
        self.port = TuntapPort(self.name, self.protocol, reactor=self.reactor, system=self.system)

    def _tunnelTypeOnly(self, flags):
        """
        Mask off any flags except for L{TunnelType.IFF_TUN} and
        L{TunnelType.IFF_TAP}.

        @param flags: Flags from L{TunnelType} to mask.
        @type flags: L{FlagConstant}

        @return: The flags given by C{flags} except the two type flags.
        @rtype: L{FlagConstant}
        """
        return flags & (TunnelFlags.IFF_TUN | TunnelFlags.IFF_TAP)

    def test_startListeningOpensDevice(self):
        """
        L{TuntapPort.startListening} opens the tunnel factory character special
        device C{"/dev/net/tun"} and configures it as a I{tun} tunnel.
        """
        system = self.system
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        expected = (system.O_RDWR | system.O_CLOEXEC | system.O_NONBLOCK, b'tun0' + b'\x00' * (_IFNAMSIZ - len(b'tun0')), self.port.interface, False, True)
        actual = (tunnel.openFlags, tunnel.requestedName, tunnel.name, tunnel.blocking, tunnel.closeOnExec)
        self.assertEqual(expected, actual)

    def test_startListeningSetsConnected(self):
        """
        L{TuntapPort.startListening} sets C{connected} on the port object to
        C{True}.
        """
        self.port.startListening()
        self.assertTrue(self.port.connected)

    def test_startListeningConnectsProtocol(self):
        """
        L{TuntapPort.startListening} calls C{makeConnection} on the protocol
        the port was initialized with, passing the port as an argument.
        """
        self.port.startListening()
        self.assertIs(self.port, self.protocol.transport)

    def test_startListeningStartsReading(self):
        """
        L{TuntapPort.startListening} passes the port instance to the reactor's
        C{addReader} method to begin watching the port's file descriptor for
        data to read.
        """
        self.port.startListening()
        self.assertIn(self.port, self.reactor.getReaders())

    def test_startListeningHandlesOpenFailure(self):
        """
        L{TuntapPort.startListening} raises L{CannotListenError} if opening the
        tunnel factory character special device fails.
        """
        self.system.permissions.remove('open')
        self.assertRaises(CannotListenError, self.port.startListening)

    def test_startListeningHandlesConfigureFailure(self):
        """
        L{TuntapPort.startListening} raises L{CannotListenError} if the
        C{ioctl} call to configure the tunnel device fails.
        """
        self.system.permissions.remove('ioctl')
        self.assertRaises(CannotListenError, self.port.startListening)

    def _stopPort(self, port):
        """
        Verify that the C{stopListening} method of an L{IListeningPort} removes
        that port from the reactor's "readers" set and also that the
        L{Deferred} returned by that method fires with L{None}.

        @param port: The port object to stop.
        @type port: L{IListeningPort} provider
        """
        stopped = port.stopListening()
        self.assertNotIn(port, self.reactor.getReaders())
        self.reactor.advance(0)
        self.assertIsNone(self.successResultOf(stopped))

    def test_stopListeningStopsReading(self):
        """
        L{TuntapPort.stopListening} returns a L{Deferred} which fires after the
        port has been removed from the reactor's reader list by passing it to
        the reactor's C{removeReader} method.
        """
        self.port.startListening()
        fileno = self.port.fileno()
        self._stopPort(self.port)
        self.assertNotIn(fileno, self.system._openFiles)

    def test_stopListeningUnsetsConnected(self):
        """
        After the L{Deferred} returned by L{TuntapPort.stopListening} fires,
        the C{connected} attribute of the port object is set to C{False}.
        """
        self.port.startListening()
        self._stopPort(self.port)
        self.assertFalse(self.port.connected)

    def test_stopListeningStopsProtocol(self):
        """
        L{TuntapPort.stopListening} calls C{doStop} on the protocol the port
        was initialized with.
        """
        self.port.startListening()
        self._stopPort(self.port)
        self.assertIsNone(self.protocol.transport)

    def test_stopListeningWhenStopped(self):
        """
        L{TuntapPort.stopListening} returns a L{Deferred} which succeeds
        immediately if it is called when the port is not listening.
        """
        stopped = self.port.stopListening()
        self.assertIsNone(self.successResultOf(stopped))

    def test_multipleStopListening(self):
        """
        It is safe and a no-op to call L{TuntapPort.stopListening} more than
        once with no intervening L{TuntapPort.startListening} call.
        """
        self.port.startListening()
        self.port.stopListening()
        second = self.port.stopListening()
        self.reactor.advance(0)
        self.assertIsNone(self.successResultOf(second))

    def test_loseConnection(self):
        """
        L{TuntapPort.loseConnection} stops the port and is deprecated.
        """
        self.port.startListening()
        self.port.loseConnection()
        self.reactor.advance(0)
        self.assertFalse(self.port.connected)
        warnings = self.flushWarnings([self.test_loseConnection])
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual('twisted.pair.tuntap.TuntapPort.loseConnection was deprecated in Twisted 14.0.0; please use twisted.pair.tuntap.TuntapPort.stopListening instead', warnings[0]['message'])
        self.assertEqual(1, len(warnings))

    def _stopsReadingTest(self, style):
        """
        Test that L{TuntapPort.doRead} has no side-effects under a certain
        exception condition.

        @param style: An exception instance to arrange for the (python wrapper
            around the) underlying platform I{read} call to fail with.

        @raise C{self.failureException}: If there are any observable
            side-effects.
        """
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.nonBlockingExceptionStyle = style
        self.port.doRead()
        self.assertEqual([], self.protocol.received)

    def test_eagainStopsReading(self):
        """
        Once L{TuntapPort.doRead} encounters an I{EAGAIN} errno from a C{read}
        call, it returns.
        """
        self._stopsReadingTest(Tunnel.EAGAIN_STYLE)

    def test_ewouldblockStopsReading(self):
        """
        Once L{TuntapPort.doRead} encounters an I{EWOULDBLOCK} errno from a
        C{read} call, it returns.
        """
        self._stopsReadingTest(Tunnel.EWOULDBLOCK_STYLE)

    def test_eintrblockStopsReading(self):
        """
        Once L{TuntapPort.doRead} encounters an I{EINTR} errno from a C{read}
        call, it returns.
        """
        self._stopsReadingTest(Tunnel.EINTR_STYLE)

    def test_unhandledReadError(self):
        """
        If L{Tuntap.doRead} encounters any exception other than one explicitly
        handled by the code, the exception propagates to the caller.
        """

        class UnexpectedException(Exception):
            pass
        self.assertRaises(UnexpectedException, self._stopsReadingTest, UnexpectedException())

    def test_unhandledEnvironmentReadError(self):
        """
        Just like C{test_unhandledReadError}, but for the case where the
        exception that is not explicitly handled happens to be of type
        C{EnvironmentError} (C{OSError} or C{IOError}).
        """
        self.assertRaises(IOError, self._stopsReadingTest, IOError(EPERM, 'Operation not permitted'))

    def test_doReadSmallDatagram(self):
        """
        L{TuntapPort.doRead} reads a datagram of fewer than
        C{TuntapPort.maxPacketSize} from the port's file descriptor and passes
        it to its protocol's C{datagramReceived} method.
        """
        datagram = b'x' * (self.port.maxPacketSize - 1)
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.append(datagram)
        self.port.doRead()
        self.assertEqual([datagram], self.protocol.received)

    def test_doReadLargeDatagram(self):
        """
        L{TuntapPort.doRead} reads the first part of a datagram of more than
        C{TuntapPort.maxPacketSize} from the port's file descriptor and passes
        the truncated data to its protocol's C{datagramReceived} method.
        """
        datagram = b'x' * self.port.maxPacketSize
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.append(datagram + b'y')
        self.port.doRead()
        self.assertEqual([datagram], self.protocol.received)

    def test_doReadSeveralDatagrams(self):
        """
        L{TuntapPort.doRead} reads several datagrams, of up to
        C{TuntapPort.maxThroughput} bytes total, before returning.
        """
        values = cycle(iterbytes(b'abcdefghijklmnopqrstuvwxyz'))
        total = 0
        datagrams = []
        while total < self.port.maxThroughput:
            datagrams.append(next(values) * self.port.maxPacketSize)
            total += self.port.maxPacketSize
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.readBuffer.extend(datagrams)
        tunnel.readBuffer.append(b'excessive datagram, not to be read')
        self.port.doRead()
        self.assertEqual(datagrams, self.protocol.received)

    def _datagramReceivedException(self):
        """
        Deliver some data to a L{TuntapPort} hooked up to an application
        protocol that raises an exception from its C{datagramReceived} method.

        @return: Whatever L{AttributeError} exceptions are logged.
        """
        self.port.startListening()
        self.system.getTunnel(self.port).readBuffer.append(b'ping')
        self.protocol.received = None
        self.port.doRead()
        return self.flushLoggedErrors(AttributeError)

    def test_datagramReceivedException(self):
        """
        If the protocol's C{datagramReceived} method raises an exception, the
        exception is logged.
        """
        errors = self._datagramReceivedException()
        self.assertEqual(1, len(errors))

    def test_datagramReceivedExceptionIdentifiesProtocol(self):
        """
        The exception raised by C{datagramReceived} is logged with a message
        identifying the offending protocol.
        """
        messages = []
        addObserver(messages.append)
        self.addCleanup(removeObserver, messages.append)
        self._datagramReceivedException()
        error = next((m for m in messages if m['isError']))
        message = textFromEventDict(error)
        self.assertEqual('Unhandled exception from %s.datagramReceived' % (fullyQualifiedName(self.protocol.__class__),), message.splitlines()[0])

    def test_write(self):
        """
        L{TuntapPort.write} sends a datagram into the tunnel.
        """
        datagram = b'a b c d e f g'
        self.port.startListening()
        self.port.write(datagram)
        self.assertEqual(self.system.getTunnel(self.port).writeBuffer, deque([datagram]))

    def test_interruptedWrite(self):
        """
        If the platform write call is interrupted (causing the Python wrapper
        to raise C{IOError} with errno set to C{EINTR}), the write is re-tried.
        """
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        tunnel.pendingSignals.append(SIGINT)
        self.port.write(b'hello, world')
        self.assertEqual(deque([b'hello, world']), tunnel.writeBuffer)

    def test_unhandledWriteError(self):
        """
        Any exception raised by the underlying write call, except for EINTR, is
        propagated to the caller.
        """
        self.port.startListening()
        tunnel = self.system.getTunnel(self.port)
        self.assertRaises(IOError, self.port.write, b'x' * tunnel.SEND_BUFFER_SIZE + b'y')

    def test_writeSequence(self):
        """
        L{TuntapPort.writeSequence} sends a datagram into the tunnel by
        concatenating the byte strings in the list passed to it.
        """
        datagram = [b'a', b'b', b'c', b'd']
        self.port.startListening()
        self.port.writeSequence(datagram)
        self.assertEqual(self.system.getTunnel(self.port).writeBuffer, deque([b''.join(datagram)]))

    def test_getHost(self):
        """
        L{TuntapPort.getHost} returns a L{TunnelAddress} including the tunnel's
        type and name.
        """
        self.port.startListening()
        address = self.port.getHost()
        self.assertEqual(TunnelAddress(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE), self.system.getTunnel(self.port).name), address)

    def test_listeningString(self):
        """
        The string representation of a L{TuntapPort} instance includes the
        tunnel type and interface and the protocol associated with the port.
        """
        self.port.startListening()
        self.assertRegex(str(self.port), fullyQualifiedName(self.protocol.__class__))
        expected = ' listening on {}/{}>'.format(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name, self.system.getTunnel(self.port).name)
        self.assertTrue(str(self.port).find(expected) != -1)

    def test_unlisteningString(self):
        """
        The string representation of a L{TuntapPort} instance includes the
        tunnel type and interface and the protocol associated with the port.
        """
        self.assertRegex(str(self.port), fullyQualifiedName(self.protocol.__class__))
        expected = ' not listening on {}/{}>'.format(self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name, self.name)
        self.assertTrue(str(self.port).find(expected) != -1)

    def test_logPrefix(self):
        """
        L{TuntapPort.logPrefix} returns a string identifying the application
        protocol and the type of tunnel.
        """
        self.assertEqual('%s (%s)' % (self.protocol.__class__.__name__, self._tunnelTypeOnly(self.helper.TUNNEL_TYPE).name), self.port.logPrefix())