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
def writeErr(self, data):
    """
        Record the extended data in the buffer.  This was an old interface
        that allowed the Transports from ISession.openShell() or
        ISession.execCommand() to receive extended data from the client.
        """
    self.err += data