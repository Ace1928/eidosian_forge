import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
class FormatFailureTests(TestCase):
    """
    Tests for L{twisted.web.util.formatFailure} which returns an HTML string
    representing the L{Failure} instance passed to it.
    """

    def test_flattenerError(self):
        """
        If there is an error flattening the L{Failure} instance,
        L{formatFailure} raises L{FlattenerError}.
        """
        self.assertRaises(FlattenerError, formatFailure, object())

    def test_returnsBytes(self):
        """
        The return value of L{formatFailure} is a C{str} instance (not a
        C{unicode} instance) with numeric character references for any non-ASCII
        characters meant to appear in the output.
        """
        try:
            raise Exception('Fake bug')
        except BaseException:
            result = formatFailure(Failure())
        self.assertIsInstance(result, bytes)
        self.assertTrue(all((ch < 128 for ch in result)))
        self.assertIn(b'&#160;', result)