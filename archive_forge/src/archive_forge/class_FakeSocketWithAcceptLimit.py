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
class FakeSocketWithAcceptLimit:
    """
            Pretend to be a socket in an overloaded system whose
            C{accept} method can only be called
            C{maximumNumberOfAccepts} times.
            """

    def accept(oself):
        acceptCalls[0] += 1
        if acceptCalls[0] > maximumNumberOfAccepts:
            self.fail('Maximum number of accept calls exceeded.')
        raise OSError(EPERM, os.strerror(EPERM))