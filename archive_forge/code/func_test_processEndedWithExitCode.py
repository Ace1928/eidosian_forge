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
def test_processEndedWithExitCode(self):
    """
        When processEnded is called, if there is an exit code in the reason
        it should be sent in an exit-status method.  The connection should be
        closed.
        """
    self.pp.processEnded(Failure(ProcessDone(None)))
    self.assertRequestsEqual([(b'exit-status', struct.pack('>I', 0), False)])
    self.assertSessionClosed()