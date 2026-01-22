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
class MyClientFactory(MyProtocolFactoryMixin, protocol.ClientFactory):
    """
    Client factory which creates L{AccumulatingProtocol} instances.
    """
    failed = 0
    stopped = 0

    def __init__(self):
        MyProtocolFactoryMixin.__init__(self)
        self.deferred = defer.Deferred()
        self.failDeferred = defer.Deferred()

    def clientConnectionFailed(self, connector, reason):
        self.failed = 1
        self.reason = reason
        self.failDeferred.callback(None)

    def clientConnectionLost(self, connector, reason):
        self.lostReason = reason
        self.deferred.callback(None)

    def stopFactory(self):
        self.stopped = 1