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
def _check1connect2(results, cf1):
    self.assertEqual(cf1.protocol.made, 1)
    d1 = defer.Deferred()
    d2 = defer.Deferred()
    port = cf1.protocol.transport.getHost().port
    cf2 = MyClientFactory()
    cf2.clientConnectionFailed = self._fireWhenDoneFunc(d1, cf2.clientConnectionFailed)
    cf2.stopFactory = self._fireWhenDoneFunc(d2, cf2.stopFactory)
    reactor.connectTCP('127.0.0.1', p.getHost().port, cf2, bindAddress=('127.0.0.1', port))
    d1.addCallback(_check2failed, cf1, cf2)
    d2.addCallback(_check2stopped, cf1, cf2)
    dl = defer.DeferredList([d1, d2])
    dl.addCallback(_stop, cf1, cf2)
    return dl