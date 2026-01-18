from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
def test_transportSemanticsToSocketType(self):
    """
        When passed a C{transportSemantics} paramter, C{'TCP'} (the value
        present in L{IPv4Address.type} to indicate a stream transport) maps to
        C{SOCK_STREAM} and C{'UDP'} maps to C{SOCK_DGRAM}.
        """
    receiver = ResultHolder(self)
    self.resolver.resolveHostName(receiver, 'example.com', transportSemantics='TCP')
    receiver2 = ResultHolder(self)
    self.resolver.resolveHostName(receiver2, 'example.com', transportSemantics='UDP')
    self.doThreadWork()
    self.doReactorWork()
    self.doThreadWork()
    self.doReactorWork()
    host, port, family, socktypeT, proto, flags = self.getter.calls[0]
    host, port, family, socktypeU, proto, flags = self.getter.calls[1]
    self.assertEqual(socktypeT, SOCK_STREAM)
    self.assertEqual(socktypeU, SOCK_DGRAM)