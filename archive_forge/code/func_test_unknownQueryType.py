from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_unknownQueryType(self):
    """
        L{ResolverBase.query} returns a L{Deferred} which fails with
        L{NotImplementedError} when called with a query of a type not present in
        its C{typeToMethod} dictionary.
        """
    resolver = ResolverBase()
    resolver.typeToMethod = {}
    query = Query(name=b'example.com', type=12345)
    queryDeferred = resolver.query(query, 123)
    result = []
    queryDeferred.addBoth(result.append)
    self.assertIsInstance(result[0], Failure)
    result[0].trap(NotImplementedError)