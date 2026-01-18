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
def test_socketTypeToAddressType(self):
    """
        When L{GAIResolver} receives a C{SOCK_DGRAM} result from
        C{getaddrinfo}, it returns a C{'TCP'} L{IPv4Address} or L{IPv6Address};
        if it receives C{SOCK_STREAM} then it returns a C{'UDP'} type of same.
        """
    receiver = ResultHolder(self)
    flowInfo = 1
    scopeID = 2
    for socktype in (SOCK_STREAM, SOCK_DGRAM):
        self.getter.addResultForHost('example.com', ('::1', 0, flowInfo, scopeID), family=AF_INET6, socktype=socktype)
        self.getter.addResultForHost('example.com', ('127.0.0.3', 0), family=AF_INET, socktype=socktype)
    self.resolver.resolveHostName(receiver, 'example.com')
    self.doThreadWork()
    self.doReactorWork()
    stream4, stream6, dgram4, dgram6 = receiver._addresses
    self.assertEqual(stream4.type, 'TCP')
    self.assertEqual(stream6.type, 'TCP')
    self.assertEqual(dgram4.type, 'UDP')
    self.assertEqual(dgram6.type, 'UDP')