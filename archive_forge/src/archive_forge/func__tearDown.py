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
def _tearDown(self, ignored):
    self.assertEqual(self.client.closed, 1)
    self.assertEqual(self.f.protocol.closed, 0)
    d = defer.Deferred()

    def _connectionLost(reason):
        self.f.protocol.closed = 1
        d.callback(None)
    self.f.protocol.connectionLost = _connectionLost
    self.f.protocol.transport.loseConnection()
    d.addCallback(lambda x: self.assertEqual(self.f.protocol.closed, 1))
    return d