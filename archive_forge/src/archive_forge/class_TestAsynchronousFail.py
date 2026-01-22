from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class TestAsynchronousFail(unittest.TestCase):
    """
    Test failures for L{unittest.TestCase} based classes.
    """
    text = 'I fail'

    def test_fail(self) -> defer.Deferred[None]:
        """
        A test which fails in the callback of the returned L{defer.Deferred}.
        """
        return deferLater(reactor, 0, self.fail, 'I fail later')

    def test_failGreaterThan64k(self) -> defer.Deferred[None]:
        """
        A test which fails in the callback of the returned L{defer.Deferred}
        with a very long string.
        """
        return deferLater(reactor, 0, self.fail, 'I fail later: ' + 'x' * 2 ** 16)

    def test_exception(self) -> None:
        """
        A test which raises an exception synchronously.
        """
        raise Exception(self.text)

    def test_exceptionGreaterThan64k(self) -> None:
        """
        A test which raises an exception with a long string representation
        synchronously.
        """
        raise LargeError(2 ** 16)

    def test_exceptionGreaterThan64kEncoded(self) -> None:
        """
        A test which synchronously raises an exception with a long string
        representation including non-ascii content.
        """
        raise Exception('☃' * 2 ** 15)