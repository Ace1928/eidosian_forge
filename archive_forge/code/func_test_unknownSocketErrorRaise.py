import errno
import os
import socket
from unittest import skipIf
from twisted.internet import interfaces, reactor
from twisted.internet.defer import gatherResults, maybeDeferred
from twisted.internet.protocol import Protocol, ServerFactory
from twisted.internet.tcp import (
from twisted.python import log
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_unknownSocketErrorRaise(self):
    """
        A C{socket.error} raised by C{accept(2)} whose C{errno} is
        unknown to the recovery logic is logged.
        """
    knownErrors = list(_ACCEPT_ERRORS)
    knownErrors.extend([EAGAIN, EPERM, EWOULDBLOCK])
    unknownAcceptError = max((error for error in knownErrors if isinstance(error, int))) + 1

    class FakeSocketWithUnknownAcceptError:
        """
            Pretend to be a socket in an overloaded system whose
            C{accept} method can only be called
            C{maximumNumberOfAccepts} times.
            """

        def accept(oself):
            raise OSError(unknownAcceptError, 'unknown socket error message')
    factory = ServerFactory()
    port = self.port(0, factory, interface='127.0.0.1')
    self.patch(port, 'socket', FakeSocketWithUnknownAcceptError())
    port.doRead()
    failures = self.flushLoggedErrors(socket.error)
    self.assertEqual(1, len(failures))
    self.assertEqual(failures[0].value.args[0], unknownAcceptError)