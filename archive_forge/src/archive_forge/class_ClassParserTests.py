import builtins
import struct
from io import StringIO
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.protocols import ident
from twisted.python import failure
from twisted.trial import unittest
class ClassParserTests(unittest.TestCase):
    """
    Test parsing of ident responses.
    """

    def setUp(self):
        """
        Create an ident client used in tests.
        """
        self.client = ident.IdentClient()

    def test_indentError(self):
        """
        'UNKNOWN-ERROR' error should map to the L{ident.IdentError} exception.
        """
        d = defer.Deferred()
        self.client.queries.append((d, 123, 456))
        self.client.lineReceived('123, 456 : ERROR : UNKNOWN-ERROR')
        return self.assertFailure(d, ident.IdentError)

    def test_noUSerError(self):
        """
        'NO-USER' error should map to the L{ident.NoUser} exception.
        """
        d = defer.Deferred()
        self.client.queries.append((d, 234, 456))
        self.client.lineReceived('234, 456 : ERROR : NO-USER')
        return self.assertFailure(d, ident.NoUser)

    def test_invalidPortError(self):
        """
        'INVALID-PORT' error should map to the L{ident.InvalidPort} exception.
        """
        d = defer.Deferred()
        self.client.queries.append((d, 345, 567))
        self.client.lineReceived('345, 567 :  ERROR : INVALID-PORT')
        return self.assertFailure(d, ident.InvalidPort)

    def test_hiddenUserError(self):
        """
        'HIDDEN-USER' error should map to the L{ident.HiddenUser} exception.
        """
        d = defer.Deferred()
        self.client.queries.append((d, 567, 789))
        self.client.lineReceived('567, 789 : ERROR : HIDDEN-USER')
        return self.assertFailure(d, ident.HiddenUser)

    def test_lostConnection(self):
        """
        A pending query which failed because of a ConnectionLost should
        receive an L{ident.IdentError}.
        """
        d = defer.Deferred()
        self.client.queries.append((d, 765, 432))
        self.client.connectionLost(failure.Failure(error.ConnectionLost()))
        return self.assertFailure(d, ident.IdentError)