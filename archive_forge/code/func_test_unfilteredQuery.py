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
def test_unfilteredQuery(self):
    """
        Similar to L{test_filteredQuery}, but for the case where a false value
        is passed for the C{filter} parameter.  In this case, the result is a
        L{Message} instance.
        """
    message = self._queryTest(False)
    self.assertIsInstance(message, Message)
    self.assertEqual(message.queries, [])
    self.assertEqual(message.answers, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
    self.assertEqual(message.authority, [])
    self.assertEqual(message.additional, [])