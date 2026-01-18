from twisted.python import usage
from twisted.trial import unittest
def test_tooManyArguments(self):
    """
        L{usage.flagFunction} raises L{usage.UsageError} if the method checked
        allows more than one argument.
        """
    exc = self.assertRaises(usage.UsageError, usage.flagFunction, self.SomeClass().manyArgs)
    self.assertEqual('Invalid Option function for manyArgs', str(exc))