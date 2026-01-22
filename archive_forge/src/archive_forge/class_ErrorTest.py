from unittest import skipIf
from twisted.internet import defer, protocol, reactor
from twisted.internet.task import deferLater
from twisted.trial import unittest, util
class ErrorTest(unittest.SynchronousTestCase):
    """
    A test case which has a L{test_foo} which will raise an error.

    @ivar ran: boolean indicating whether L{test_foo} has been run.
    """
    ran = False

    def test_foo(self):
        """
        Set C{self.ran} to True and raise a C{ZeroDivisionError}
        """
        self.ran = True
        1 / 0