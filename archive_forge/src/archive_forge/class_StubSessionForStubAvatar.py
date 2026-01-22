import os
import signal
import struct
import sys
from unittest import skipIf
from zope.interface import implementer
from twisted.internet import defer, error, protocol
from twisted.internet.address import IPv4Address
from twisted.internet.error import ProcessDone, ProcessTerminated
from twisted.python import components, failure
from twisted.python.failure import Failure
from twisted.python.reflect import requireModule
from twisted.python.test.test_components import RegistryUsingMixin
from twisted.trial.unittest import TestCase
@implementer(session.ISession)
class StubSessionForStubAvatar:
    """
    A stub ISession implementation for our StubAvatar.  The instance
    variables generally keep track of method invocations so that we can test
    that the methods were called.

    @ivar avatar: the L{StubAvatar} we are adapting.
    @ivar ptyRequest: if present, the terminal, window size, and modes passed
        to the getPty method.
    @ivar windowChange: if present, the window size passed to the
        windowChangned method.
    @ivar shellProtocol: if present, the L{SSHSessionProcessProtocol} passed
        to the openShell method.
    @ivar shellTransport: if present, the L{EchoTransport} connected to
        shellProtocol.
    @ivar execProtocol: if present, the L{SSHSessionProcessProtocol} passed
        to the execCommand method.
    @ivar execTransport: if present, the L{EchoTransport} connected to
        execProtocol.
    @ivar execCommandLine: if present, the command line passed to the
        execCommand method.
    @ivar gotEOF: if present, an EOF message was received.
    @ivar gotClosed: if present, a closed message was received.
    """

    def __init__(self, avatar):
        """
        Store the avatar we're adapting.
        """
        self.avatar = avatar
        self.shellProtocol = None

    def getPty(self, terminal, window, modes):
        """
        If the terminal is 'bad', fail.  Otherwise, store the information in
        the ptyRequest variable.
        """
        if terminal != b'bad':
            self.ptyRequest = (terminal, window, modes)
        else:
            raise RuntimeError('not getting a pty')

    def windowChanged(self, window):
        """
        If all the window sizes are 0, fail.  Otherwise, store the size in the
        windowChange variable.
        """
        if window == (0, 0, 0, 0):
            raise RuntimeError('not changing the window size')
        else:
            self.windowChange = window

    def openShell(self, pp):
        """
        If we have gotten a shell request before, fail.  Otherwise, store the
        process protocol in the shellProtocol variable, connect it to the
        EchoTransport and store that as shellTransport.
        """
        if self.shellProtocol is not None:
            raise RuntimeError('not getting a shell this time')
        else:
            self.shellProtocol = pp
            self.shellTransport = EchoTransport(pp)

    def execCommand(self, pp, command):
        """
        If the command is 'true', store the command, the process protocol, and
        the transport we connect to the process protocol.  Otherwise, just
        store the command and raise an error.
        """
        self.execCommandLine = command
        if command == b'success':
            self.execProtocol = pp
        elif command[:6] == b'repeat':
            self.execProtocol = pp
            self.execTransport = EchoTransport(pp)
            pp.outReceived(command[7:])
        else:
            raise RuntimeError('not getting a command')

    def eofReceived(self):
        """
        Note that EOF has been received.
        """
        self.gotEOF = True

    def closed(self):
        """
        Note that close has been received.
        """
        self.gotClosed = True