from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class EventuallyFailingTestCase(unittest.SynchronousTestCase):
    """
    A test suite that fails after it is run a few times.
    """
    n: int = 0

    def test_it(self):
        """
        Run successfully a few times and then fail forever after.
        """
        self.n += 1
        if self.n >= 5:
            self.fail('eventually failing')