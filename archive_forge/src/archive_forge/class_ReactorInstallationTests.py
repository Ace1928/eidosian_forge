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
class ReactorInstallationTests(UnitTest):
    """
    Tests for installing old and new resolvers onto a
    L{PluggableResolverMixin} and L{ReactorBase} (from which all of Twisted's
    reactor implementations derive).
    """

    def test_interfaceCompliance(self):
        """
        L{PluggableResolverMixin} (and its subclasses) implement both
        L{IReactorPluggableNameResolver} and L{IReactorPluggableResolver}.
        """
        reactor = PluggableResolverMixin()
        verifyObject(IReactorPluggableNameResolver, reactor)
        verifyObject(IResolverSimple, reactor.resolver)
        verifyObject(IHostnameResolver, reactor.nameResolver)

    def test_installingOldStyleResolver(self):
        """
        L{PluggableResolverMixin} will wrap an L{IResolverSimple} in a
        complexifier.
        """
        reactor = PluggableResolverMixin()
        it = SillyResolverSimple()
        verifyObject(IResolverSimple, reactor.installResolver(it))
        self.assertIsInstance(reactor.nameResolver, SimpleResolverComplexifier)
        self.assertIs(reactor.nameResolver._simpleResolver, it)

    def test_defaultToGAIResolver(self):
        """
        L{ReactorBase} defaults to using a L{GAIResolver}.
        """
        reactor = JustEnoughReactor()
        self.assertIsInstance(reactor.nameResolver, GAIResolver)
        self.assertIs(reactor.nameResolver._getaddrinfo, getaddrinfo)
        self.assertIsInstance(reactor.resolver, ComplexResolverSimplifier)
        self.assertIs(reactor.nameResolver._reactor, reactor)
        self.assertIs(reactor.resolver._nameResolver, reactor.nameResolver)