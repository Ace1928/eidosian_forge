from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class FailureButTearDownRunsMixin:
    """
    A test fails, but its L{tearDown} still runs.
    """
    tornDown = False

    def tearDown(self):
        self.tornDown = True

    def test_fails(self):
        """
        A test that fails.
        """
        raise FoolishError('I am a broken test')