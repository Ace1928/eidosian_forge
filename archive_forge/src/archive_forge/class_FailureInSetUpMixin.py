from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class FailureInSetUpMixin:

    def setUp(self):
        raise FoolishError('I am a broken setUp method')

    def test_noop(self):
        pass