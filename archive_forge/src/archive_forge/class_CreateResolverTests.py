import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
class CreateResolverTests(unittest.TestCase, GoodTempPathMixin):
    """
    Tests for L{client.createResolver}.
    """
    skip = windowsSkip

    def _hostsTest(self, resolver, filename):
        res = [r for r in resolver.resolvers if isinstance(r, hosts.Resolver)]
        self.assertEqual(1, len(res))
        self.assertEqual(res[0].file, filename)

    def test_defaultHosts(self):
        """
        L{client.createResolver} returns a L{resolve.ResolverChain} including a
        L{hosts.Resolver} using I{/etc/hosts} if no alternate hosts file is
        specified.
        """
        with AlternateReactor(Clock()):
            resolver = client.createResolver()
        self._hostsTest(resolver, b'/etc/hosts')

    def test_overrideHosts(self):
        """
        The I{hosts} parameter to L{client.createResolver} overrides the hosts
        file used by the L{hosts.Resolver} in the L{resolve.ResolverChain} it
        returns.
        """
        with AlternateReactor(Clock()):
            resolver = client.createResolver(hosts=b'/foo/bar')
        self._hostsTest(resolver, b'/foo/bar')

    def _resolvConfTest(self, resolver, filename):
        """
        Verify that C{resolver} has a L{client.Resolver} with a configuration
        filename set to C{filename}.
        """
        res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
        self.assertEqual(1, len(res))
        self.assertEqual(res[0].resolv, filename)

    def test_reactor(self):
        """
        The L{client.Resolver} included in the L{resolve.ResolverChain} returned
        by L{client.createResolver} uses the global reactor.
        """
        reactor = Clock()
        with AlternateReactor(reactor):
            resolver = client.createResolver()
        res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
        self.assertEqual(1, len(res))
        self.assertIs(reactor, res[0]._reactor)

    def test_defaultResolvConf(self):
        """
        L{client.createResolver} returns a L{resolve.ResolverChain} including a
        L{client.Resolver} using I{/etc/resolv.conf} if no alternate resolver
        configuration file is specified.
        """
        with AlternateReactor(Clock()):
            resolver = client.createResolver()
        self._resolvConfTest(resolver, b'/etc/resolv.conf')

    def test_overrideResolvConf(self):
        """
        The I{resolvconf} parameter to L{client.createResolver} overrides the
        resolver configuration file used by the L{client.Resolver} in the
        L{resolve.ResolverChain} it returns.
        """
        with AlternateReactor(Clock()):
            resolver = client.createResolver(resolvconf=b'/foo/bar')
        self._resolvConfTest(resolver, b'/foo/bar')

    def test_defaultServers(self):
        """
        If no servers are given, addresses are taken from the file given by the
        I{resolvconf} parameter to L{client.createResolver}.
        """
        resolvconf = self.path()
        resolvconf.setContent(b'nameserver 127.1.2.3\n')
        with AlternateReactor(Clock()):
            resolver = client.createResolver(resolvconf=resolvconf.path)
        res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
        self.assertEqual(1, len(res))
        self.assertEqual([], res[0].servers)
        self.assertEqual([('127.1.2.3', 53)], res[0].dynServers)

    def test_overrideServers(self):
        """
        Servers passed to L{client.createResolver} are used in addition to any
        found in the file given by the I{resolvconf} parameter.
        """
        resolvconf = self.path()
        resolvconf.setContent(b'nameserver 127.1.2.3\n')
        with AlternateReactor(Clock()):
            resolver = client.createResolver(servers=[('127.3.2.1', 53)], resolvconf=resolvconf.path)
        res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
        self.assertEqual(1, len(res))
        self.assertEqual([('127.3.2.1', 53)], res[0].servers)
        self.assertEqual([('127.1.2.3', 53)], res[0].dynServers)

    def test_cache(self):
        """
        L{client.createResolver} returns a L{resolve.ResolverChain} including a
        L{cache.CacheResolver}.
        """
        with AlternateReactor(Clock()):
            resolver = client.createResolver()
        res = [r for r in resolver.resolvers if isinstance(r, cache.CacheResolver)]
        self.assertEqual(1, len(res))