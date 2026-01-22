from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
class RootResolverResolverFactoryTests(TestCase):
    """
    Tests for L{root.Resolver._resolverFactory}.
    """

    def test_resolverFactoryArgumentPresent(self):
        """
        L{root.Resolver.__init__} accepts a C{resolverFactory}
        argument and assigns it to C{self._resolverFactory}.
        """
        r = Resolver(hints=[None], resolverFactory=raisingResolverFactory)
        self.assertIs(r._resolverFactory, raisingResolverFactory)

    def test_resolverFactoryArgumentAbsent(self):
        """
        L{root.Resolver.__init__} sets L{client.Resolver} as the
        C{_resolverFactory} if a C{resolverFactory} argument is not
        supplied.
        """
        r = Resolver(hints=[None])
        self.assertIs(r._resolverFactory, client.Resolver)

    def test_resolverFactoryOnlyExpectedArguments(self):
        """
        L{root.Resolver._resolverFactory} is supplied with C{reactor} and
        C{servers} keyword arguments.
        """
        dummyReactor = object()
        r = Resolver(hints=['192.0.2.101'], resolverFactory=raisingResolverFactory, reactor=dummyReactor)
        e = self.assertRaises(ResolverFactoryArguments, r.lookupAddress, 'example.com')
        self.assertEqual(((), {'reactor': dummyReactor, 'servers': [('192.0.2.101', 53)]}), (e.args, e.kwargs))