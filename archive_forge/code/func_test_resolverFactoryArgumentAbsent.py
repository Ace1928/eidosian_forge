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
def test_resolverFactoryArgumentAbsent(self):
    """
        L{root.Resolver.__init__} sets L{client.Resolver} as the
        C{_resolverFactory} if a C{resolverFactory} argument is not
        supplied.
        """
    r = Resolver(hints=[None])
    self.assertIs(r._resolverFactory, client.Resolver)