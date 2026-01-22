import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
class ClosingFactory(protocol.ServerFactory):
    """
    Factory that closes port immediately.
    """
    _cleanerUpper = None

    def buildProtocol(self, conn):
        self._cleanerUpper = self.port.stopListening()
        return ClosingProtocol()

    def cleanUp(self):
        """
        Clean-up for tests to wait for the port to stop listening.
        """
        if self._cleanerUpper is None:
            return self.port.stopListening()
        return self._cleanerUpper