from __future__ import annotations
from zope.interface import implementer, verify
from twisted.internet import defer, interfaces
from twisted.trial import unittest
from twisted.web import client
class HTTPConnectionPoolTests(unittest.TestCase):
    """
    Unit tests for L{client.HTTPConnectionPoolTest}.
    """

    def test_implements(self) -> None:
        """L{DummyEndPoint}s implements L{interfaces.IStreamClientEndpoint}"""
        ep = DummyEndPoint('something')
        verify.verifyObject(interfaces.IStreamClientEndpoint, ep)

    def test_repr(self) -> None:
        """connection L{repr()} includes endpoint's L{repr()}"""
        pool = client.HTTPConnectionPool(reactor=None)
        ep = DummyEndPoint('this_is_probably_unique')
        d = pool.getConnection('someplace', ep)
        result = self.successResultOf(d)
        representation = repr(result)
        self.assertIn(repr(ep), representation)