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
class StubClient:
    """
    A stub class representing the client to a SSHSession.

    @ivar transport: A L{StubTransport} object which keeps track of the data
        passed to it.
    """

    def __init__(self):
        self.transport = StubTransportWithWriteErr()