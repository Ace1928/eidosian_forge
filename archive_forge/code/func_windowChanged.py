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
def windowChanged(self, window):
    """
        If all the window sizes are 0, fail.  Otherwise, store the size in the
        windowChange variable.
        """
    if window == (0, 0, 0, 0):
        raise RuntimeError('not changing the window size')
    else:
        self.windowChange = window