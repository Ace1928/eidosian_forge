from twisted.python import usage
from twisted.trial import unittest
class WrongTypedTests(unittest.TestCase):
    """
    Test L{usage.Options.parseOptions} for wrong coerce options.
    """

    def test_nonCallable(self):
        """
        Using a non-callable type fails.
        """
        us = WrongTypedOptions()
        argV = '--barwrong egg'.split()
        self.assertRaises(TypeError, us.parseOptions, argV)

    def test_notCalledInDefault(self):
        """
        The coerce functions are not called if no values are provided.
        """
        us = WeirdCallableOptions()
        argV = []
        us.parseOptions(argV)

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