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
def _notificationRaisesTest(self):
    """
        Helper for testing that an exception is logged by the time the
        client protocol loses its connection.
        """
    closed = self.clientProtocol.closedDeferred = defer.Deferred()
    self.clientProtocol.transport.loseWriteConnection()

    def check(ignored):
        errors = self.flushLoggedErrors(RuntimeError)
        self.assertEqual(len(errors), 1)
    closed.addCallback(check)
    return closed