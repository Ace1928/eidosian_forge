from twisted.python import usage
from twisted.trial import unittest
def test_weirdCallable(self):
    """
        Errors raised by coerce functions are handled properly.
        """
    us = WeirdCallableOptions()
    argV = '--foowrong blah'.split()
    e = self.assertRaises(usage.UsageError, us.parseOptions, argV)
    self.assertEqual(str(e), 'Parameter type enforcement failed: Yay')
    us = WeirdCallableOptions()
    argV = '--barwrong blah'.split()
    self.assertRaises(RuntimeError, us.parseOptions, argV)