from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def test_unjelliedFailureCheck(self):
    """
        An unjellied L{CopyableFailure} has a check method which behaves the
        same way as the original L{CopyableFailure}'s check method.
        """
    original = pb.CopyableFailure(ZeroDivisionError())
    self.assertIs(original.check(ZeroDivisionError), ZeroDivisionError)
    self.assertIs(original.check(ArithmeticError), ArithmeticError)
    copied = jelly.unjelly(jelly.jelly(original, invoker=DummyInvoker()))
    self.assertIs(copied.check(ZeroDivisionError), ZeroDivisionError)
    self.assertIs(copied.check(ArithmeticError), ArithmeticError)