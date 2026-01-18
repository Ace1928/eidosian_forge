from twisted.python import usage
from twisted.trial import unittest
def test_tooManyArgumentsAndSpecificErrorMessage(self):
    """
        L{usage.flagFunction} uses the given method name in the error message
        raised when the method allows too many arguments.
        """
    exc = self.assertRaises(usage.UsageError, usage.flagFunction, self.SomeClass().manyArgs, 'flubuduf')
    self.assertEqual('Invalid Option function for flubuduf', str(exc))