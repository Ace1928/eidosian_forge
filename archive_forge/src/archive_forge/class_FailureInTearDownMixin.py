from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class FailureInTearDownMixin:

    def tearDown(self):
        raise FoolishError('I am a broken tearDown method')

    def test_noop(self):
        pass