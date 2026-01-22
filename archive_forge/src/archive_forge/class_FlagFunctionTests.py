from twisted.python import usage
from twisted.trial import unittest
class FlagFunctionTests(unittest.TestCase):
    """
    Tests for L{usage.flagFunction}.
    """

    class SomeClass:
        """
        Dummy class for L{usage.flagFunction} tests.
        """

        def oneArg(self, a):
            """
            A one argument method to be tested by L{usage.flagFunction}.

            @param a: a useless argument to satisfy the function's signature.
            """

        def noArg(self):
            """
            A no argument method to be tested by L{usage.flagFunction}.
            """

        def manyArgs(self, a, b, c):
            """
            A multiple arguments method to be tested by L{usage.flagFunction}.

            @param a: a useless argument to satisfy the function's signature.
            @param b: a useless argument to satisfy the function's signature.
            @param c: a useless argument to satisfy the function's signature.
            """

    def test_hasArg(self):
        """
        L{usage.flagFunction} returns C{False} if the method checked allows
        exactly one argument.
        """
        self.assertIs(False, usage.flagFunction(self.SomeClass().oneArg))

    def test_noArg(self):
        """
        L{usage.flagFunction} returns C{True} if the method checked allows
        exactly no argument.
        """
        self.assertIs(True, usage.flagFunction(self.SomeClass().noArg))

    def test_tooManyArguments(self):
        """
        L{usage.flagFunction} raises L{usage.UsageError} if the method checked
        allows more than one argument.
        """
        exc = self.assertRaises(usage.UsageError, usage.flagFunction, self.SomeClass().manyArgs)
        self.assertEqual('Invalid Option function for manyArgs', str(exc))

    def test_tooManyArgumentsAndSpecificErrorMessage(self):
        """
        L{usage.flagFunction} uses the given method name in the error message
        raised when the method allows too many arguments.
        """
        exc = self.assertRaises(usage.UsageError, usage.flagFunction, self.SomeClass().manyArgs, 'flubuduf')
        self.assertEqual('Invalid Option function for flubuduf', str(exc))