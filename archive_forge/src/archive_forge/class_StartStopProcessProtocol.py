import errno
import os
import sys
import warnings
from os import close, pathsep, pipe, read
from socket import AF_INET, AF_INET6, SOL_SOCKET, error, socket
from struct import pack
from unittest import skipIf
from twisted.internet import reactor
from twisted.internet.defer import Deferred, inlineCallbacks
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
class StartStopProcessProtocol(ProcessProtocol):
    """
    An L{IProcessProtocol} with a Deferred for events where the subprocess
    starts and stops.

    @ivar started: A L{Deferred} which fires with this protocol's
        L{IProcessTransport} provider when it is connected to one.

    @ivar stopped: A L{Deferred} which fires with the process output or a
        failure if the process produces output on standard error.

    @ivar output: A C{str} used to accumulate standard output.

    @ivar errors: A C{str} used to accumulate standard error.
    """

    def __init__(self):
        self.started = Deferred()
        self.stopped = Deferred()
        self.output = b''
        self.errors = b''

    def connectionMade(self):
        self.started.callback(self.transport)

    def outReceived(self, data):
        self.output += data

    def errReceived(self, data):
        self.errors += data

    def processEnded(self, reason):
        if reason.check(ProcessDone):
            self.stopped.callback(self.output)
        else:
            self.stopped.errback(ExitedWithStderr(self.errors, self.output))