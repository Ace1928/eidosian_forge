from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_typeToMethodDispatch(self):
    """
        L{ResolverBase.query} looks up a method to invoke using the type of the
        query passed to it and the C{typeToMethod} mapping on itself.
        """
    results = []
    resolver = ResolverBase()
    resolver.typeToMethod = {12345: lambda query, timeout: results.append((query, timeout))}
    query = Query(name=b'example.com', type=12345)
    resolver.query(query, 123)
    self.assertEqual([(b'example.com', 123)], results)