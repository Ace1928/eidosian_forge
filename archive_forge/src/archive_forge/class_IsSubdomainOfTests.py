import struct
from io import BytesIO
from zope.interface.verify import verifyClass
from twisted.internet import address, task
from twisted.internet.error import CannotListenError, ConnectionDone
from twisted.names import dns
from twisted.python.failure import Failure
from twisted.python.util import FancyEqMixin, FancyStrMixin
from twisted.test import proto_helpers
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial import unittest
class IsSubdomainOfTests(unittest.SynchronousTestCase):
    """
    Tests for L{twisted.names.dns._isSubdomainOf}.
    """

    def test_identical(self):
        """
        L{dns._isSubdomainOf} returns C{True} for identical
        domain names.
        """
        assertIsSubdomainOf(self, b'example.com', b'example.com')

    def test_parent(self):
        """
        L{dns._isSubdomainOf} returns C{True} when the first
        name is an immediate descendant of the second name.
        """
        assertIsSubdomainOf(self, b'foo.example.com', b'example.com')

    def test_distantAncestor(self):
        """
        L{dns._isSubdomainOf} returns C{True} when the first
        name is a distant descendant of the second name.
        """
        assertIsSubdomainOf(self, b'foo.bar.baz.example.com', b'com')

    def test_superdomain(self):
        """
        L{dns._isSubdomainOf} returns C{False} when the first
        name is an ancestor of the second name.
        """
        assertIsNotSubdomainOf(self, b'example.com', b'foo.example.com')

    def test_sibling(self):
        """
        L{dns._isSubdomainOf} returns C{False} if the first name
        is a sibling of the second name.
        """
        assertIsNotSubdomainOf(self, b'foo.example.com', b'bar.example.com')

    def test_unrelatedCommonSuffix(self):
        """
        L{dns._isSubdomainOf} returns C{False} even when domain
        names happen to share a common suffix.
        """
        assertIsNotSubdomainOf(self, b'foo.myexample.com', b'example.com')

    def test_subdomainWithTrailingDot(self):
        """
        L{dns._isSubdomainOf} returns C{True} if the first name
        is a subdomain of the second name but the first name has a
        trailing ".".
        """
        assertIsSubdomainOf(self, b'foo.example.com.', b'example.com')

    def test_superdomainWithTrailingDot(self):
        """
        L{dns._isSubdomainOf} returns C{True} if the first name
        is a subdomain of the second name but the second name has a
        trailing ".".
        """
        assertIsSubdomainOf(self, b'foo.example.com', b'example.com.')

    def test_bothWithTrailingDot(self):
        """
        L{dns._isSubdomainOf} returns C{True} if the first name
        is a subdomain of the second name and both names have a
        trailing ".".
        """
        assertIsSubdomainOf(self, b'foo.example.com.', b'example.com.')

    def test_emptySubdomain(self):
        """
        L{dns._isSubdomainOf} returns C{False} if the first name
        is empty and the second name is not.
        """
        assertIsNotSubdomainOf(self, b'', b'example.com')

    def test_emptySuperdomain(self):
        """
        L{dns._isSubdomainOf} returns C{True} if the second name
        is empty and the first name is not.
        """
        assertIsSubdomainOf(self, b'foo.example.com', b'')

    def test_caseInsensitiveComparison(self):
        """
        L{dns._isSubdomainOf} does case-insensitive comparison
        of name labels.
        """
        assertIsSubdomainOf(self, b'foo.example.com', b'EXAMPLE.COM')
        assertIsSubdomainOf(self, b'FOO.EXAMPLE.COM', b'example.com')