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
def test_requestShellGetsSession(self):
    """
        If an ISession adapter isn't already present, request_shell should get
        one.
        """
    self.session.requestReceived(b'shell', b'')
    self.assertSessionProvidesISession()