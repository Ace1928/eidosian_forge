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
def test_resolverReceivesRootHints(self):
    """
        The L{root.Resolver} which eventually replaces L{root.DeferredResolver}
        is supplied with the IP addresses of the 13 root servers.
        """
    stubResolver = StubResolver()
    deferredResolver = root.bootstrap(stubResolver)
    for d in stubResolver.pendingResults:
        d.callback('192.0.2.101')
    self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 13)