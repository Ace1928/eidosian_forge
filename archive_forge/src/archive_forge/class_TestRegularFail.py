from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class TestRegularFail(unittest.SynchronousTestCase):

    def test_fail(self):
        self.fail('I fail')

    def test_subfail(self):
        self.subroutine()

    def subroutine(self):
        self.fail('I fail inside')