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
class LargeBufferTests(TestCase):
    """Test that buffering large amounts of data works."""
    datalen = 60 * 1024 * 1024

    def testWriter(self):
        f = protocol.Factory()
        f.protocol = LargeBufferWriterProtocol
        f.done = 0
        f.problem = 0
        f.len = self.datalen
        wrappedF = FireOnCloseFactory(f)
        p = reactor.listenTCP(0, wrappedF, interface='127.0.0.1')
        self.addCleanup(p.stopListening)
        n = p.getHost().port
        clientF = LargeBufferReaderClientFactory()
        wrappedClientF = FireOnCloseFactory(clientF)
        reactor.connectTCP('127.0.0.1', n, wrappedClientF)
        d = defer.gatherResults([wrappedF.deferred, wrappedClientF.deferred])

        def check(ignored):
            self.assertTrue(f.done, "writer didn't finish, it probably died")
            self.assertTrue(clientF.len >= self.datalen, "client didn't receive all the data it expected (%d != %d)" % (clientF.len, self.datalen))
            self.assertTrue(clientF.len <= self.datalen, 'client did receive more data than it expected (%d != %d)' % (clientF.len, self.datalen))
            self.assertTrue(clientF.done, "client didn't see connection dropped")
        return d.addCallback(check)