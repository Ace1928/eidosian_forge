import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
class MutualArgumentExclusionTests(SynchronousTestCase):
    """
    Tests for L{mutuallyExclusiveArguments}.
    """

    def checkPassed(self, func, *args, **kw):
        """
        Test an invocation of L{passed} with the given function, arguments, and
        keyword arguments.

        @param func: A function whose argspec will be inspected.
        @type func: A callable.

        @param args: The arguments which could be passed to C{func}.

        @param kw: The keyword arguments which could be passed to C{func}.

        @return: L{_passedSignature} or L{_passedArgSpec}'s return value
        @rtype: L{dict}
        """
        if getattr(inspect, 'signature', None):
            return _passedSignature(inspect.signature(func), args, kw)
        else:
            return _passedArgSpec(inspect.getargspec(func), args, kw)

    def test_passed_simplePositional(self):
        """
        L{passed} identifies the arguments passed by a simple
        positional test.
        """

        def func(a, b):
            pass
        self.assertEqual(self.checkPassed(func, 1, 2), dict(a=1, b=2))

    def test_passed_tooManyArgs(self):
        """
        L{passed} raises a L{TypeError} if too many arguments are
        passed.
        """

        def func(a, b):
            pass
        self.assertRaises(TypeError, self.checkPassed, func, 1, 2, 3)

    def test_passed_doublePassKeyword(self):
        """
        L{passed} raises a L{TypeError} if a argument is passed both
        positionally and by keyword.
        """

        def func(a):
            pass
        self.assertRaises(TypeError, self.checkPassed, func, 1, a=2)

    def test_passed_unspecifiedKeyword(self):
        """
        L{passed} raises a L{TypeError} if a keyword argument not
        present in the function's declaration is passed.
        """

        def func(a):
            pass
        self.assertRaises(TypeError, self.checkPassed, func, 1, z=2)

    def test_passed_star(self):
        """
        L{passed} places additional positional arguments into a tuple
        under the name of the star argument.
        """

        def func(a, *b):
            pass
        self.assertEqual(self.checkPassed(func, 1, 2, 3), dict(a=1, b=(2, 3)))

    def test_passed_starStar(self):
        """
        Additional keyword arguments are passed as a dict to the star star
        keyword argument.
        """

        def func(a, **b):
            pass
        self.assertEqual(self.checkPassed(func, 1, x=2, y=3, z=4), dict(a=1, b=dict(x=2, y=3, z=4)))

    def test_passed_noDefaultValues(self):
        """
        The results of L{passed} only include arguments explicitly
        passed, not default values.
        """

        def func(a, b, c=1, d=2, e=3):
            pass
        self.assertEqual(self.checkPassed(func, 1, 2, e=7), dict(a=1, b=2, e=7))

    def test_mutualExclusionPrimeDirective(self):
        """
        L{mutuallyExclusiveArguments} does not interfere in its
        decoratee's operation, either its receipt of arguments or its return
        value.
        """

        @_mutuallyExclusiveArguments([('a', 'b')])
        def func(x, y, a=3, b=4):
            return x + y + a + b
        self.assertEqual(func(1, 2), 10)
        self.assertEqual(func(1, 2, 7), 14)
        self.assertEqual(func(1, 2, b=7), 13)

    def test_mutualExclusionExcludesByKeyword(self):
        """
        L{mutuallyExclusiveArguments} raises a L{TypeError}n if its
        decoratee is passed a pair of mutually exclusive arguments.
        """

        @_mutuallyExclusiveArguments([['a', 'b']])
        def func(a=3, b=4):
            return a + b
        self.assertRaises(TypeError, func, a=3, b=4)

    def test_invalidParameterType(self):
        """
        Create a fake signature with an invalid parameter
        type to test error handling.  The valid parameter
        types are specified in L{inspect.Parameter}.
        """

        class FakeSignature:

            def __init__(self, parameters):
                self.parameters = parameters

        class FakeParameter:

            def __init__(self, name, kind):
                self.name = name
                self.kind = kind

        def func(a, b):
            pass
        func(1, 2)
        parameters = inspect.signature(func).parameters
        dummyParameters = parameters.copy()
        dummyParameters['c'] = FakeParameter('fake', 'fake')
        fakeSig = FakeSignature(dummyParameters)
        self.assertRaises(TypeError, _passedSignature, fakeSig, (1, 2), {})