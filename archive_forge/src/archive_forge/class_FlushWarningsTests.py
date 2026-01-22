from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
class FlushWarningsTests(SynchronousTestCase):
    """
    Tests for C{flushWarnings}, an API for examining the warnings
    emitted so far in a test.
    """

    def assertDictSubset(self, set: Mapping[_K, _V], subset: Mapping[_K, _V]) -> None:
        """
        Assert that all the keys present in C{subset} are also present in
        C{set} and that the corresponding values are equal.
        """
        for k, v in subset.items():
            self.assertEqual(set[k], v)

    def assertDictSubsets(self, sets: Sequence[Mapping[_K, _V]], subsets: Sequence[Mapping[_K, _V]]) -> None:
        """
        For each pair of corresponding elements in C{sets} and C{subsets},
        assert that the element from C{subsets} is a subset of the element from
        C{sets}.
        """
        self.assertEqual(len(sets), len(subsets))
        for a, b in zip(sets, subsets):
            self.assertDictSubset(a, b)

    def test_none(self) -> None:
        """
        If no warnings are emitted by a test, C{flushWarnings} returns an empty
        list.
        """
        self.assertEqual(self.flushWarnings(), [])

    def test_several(self) -> None:
        """
        If several warnings are emitted by a test, C{flushWarnings} returns a
        list containing all of them.
        """
        firstMessage = 'first warning message'
        firstCategory = UserWarning
        warnings.warn(message=firstMessage, category=firstCategory)
        secondMessage = 'second warning message'
        secondCategory = RuntimeWarning
        warnings.warn(message=secondMessage, category=secondCategory)
        self.assertDictSubsets(self.flushWarnings(), [{'category': firstCategory, 'message': firstMessage}, {'category': secondCategory, 'message': secondMessage}])

    def test_repeated(self) -> None:
        """
        The same warning triggered twice from the same place is included twice
        in the list returned by C{flushWarnings}.
        """
        message = 'the message'
        category = RuntimeWarning
        for i in range(2):
            warnings.warn(message=message, category=category)
        self.assertDictSubsets(self.flushWarnings(), [{'category': category, 'message': message}] * 2)

    def test_cleared(self) -> None:
        """
        After a particular warning event has been returned by C{flushWarnings},
        it is not returned by subsequent calls.
        """
        message = 'the message'
        category = RuntimeWarning
        warnings.warn(message=message, category=category)
        self.assertDictSubsets(self.flushWarnings(), [{'category': category, 'message': message}])
        self.assertEqual(self.flushWarnings(), [])

    def test_unflushed(self) -> None:
        """
        Any warnings emitted by a test which are not flushed are emitted to the
        Python warning system.
        """
        result = TestResult()
        case = Mask.MockTests('test_unflushed')
        case.run(result)
        warningsShown = self.flushWarnings([Mask.MockTests.test_unflushed])
        self.assertEqual(warningsShown[0]['message'], 'some warning text')
        self.assertIdentical(warningsShown[0]['category'], UserWarning)
        where = type(case).test_unflushed.__code__
        filename = where.co_filename
        lineno = where.co_firstlineno + 4
        self.assertEqual(warningsShown[0]['filename'], filename)
        self.assertEqual(warningsShown[0]['lineno'], lineno)
        self.assertEqual(len(warningsShown), 1)

    def test_flushed(self) -> None:
        """
        Any warnings emitted by a test which are flushed are not emitted to the
        Python warning system.
        """
        result = TestResult()
        case = Mask.MockTests('test_flushed')
        output = StringIO()
        monkey = self.patch(sys, 'stdout', output)
        case.run(result)
        monkey.restore()
        self.assertEqual(output.getvalue(), '')

    def test_warningsConfiguredAsErrors(self) -> None:
        """
        If a warnings filter has been installed which turns warnings into
        exceptions, tests have an error added to the reporter for them for each
        unflushed warning.
        """

        class CustomWarning(Warning):
            pass
        result = TestResult()
        case = Mask.MockTests('test_unflushed')
        case.category = CustomWarning
        originalWarnings = warnings.filters[:]
        try:
            warnings.simplefilter('error')
            case.run(result)
            self.assertEqual(len(result.errors), 1)
            self.assertIdentical(result.errors[0][0], case)
            self.assertTrue(result.errors[0][1].splitlines()[-1].endswith('CustomWarning: some warning text'))
        finally:
            warnings.filters[:] = originalWarnings

    def test_flushedWarningsConfiguredAsErrors(self) -> None:
        """
        If a warnings filter has been installed which turns warnings into
        exceptions, tests which emit those warnings but flush them do not have
        an error added to the reporter.
        """

        class CustomWarning(Warning):
            pass
        result = TestResult()
        case = Mask.MockTests('test_flushed')
        case.category = CustomWarning
        originalWarnings = warnings.filters[:]
        try:
            warnings.simplefilter('error')
            case.run(result)
            self.assertEqual(result.errors, [])
        finally:
            warnings.filters[:] = originalWarnings

    def test_multipleFlushes(self) -> None:
        """
        Any warnings emitted after a call to C{flushWarnings} can be flushed by
        another call to C{flushWarnings}.
        """
        warnings.warn('first message')
        self.assertEqual(len(self.flushWarnings()), 1)
        warnings.warn('second message')
        self.assertEqual(len(self.flushWarnings()), 1)

    def test_filterOnOffendingFunction(self) -> None:
        """
        The list returned by C{flushWarnings} includes only those
        warnings which refer to the source of the function passed as the value
        for C{offendingFunction}, if a value is passed for that parameter.
        """
        firstMessage = 'first warning text'
        firstCategory = UserWarning

        def one() -> None:
            warnings.warn(firstMessage, firstCategory, stacklevel=1)
        secondMessage = 'some text'
        secondCategory = RuntimeWarning

        def two() -> None:
            warnings.warn(secondMessage, secondCategory, stacklevel=1)
        one()
        two()
        self.assertDictSubsets(self.flushWarnings(offendingFunctions=[one]), [{'category': firstCategory, 'message': firstMessage}])
        self.assertDictSubsets(self.flushWarnings(offendingFunctions=[two]), [{'category': secondCategory, 'message': secondMessage}])

    def test_functionBoundaries(self) -> None:
        """
        Verify that warnings emitted at the very edges of a function are still
        determined to be emitted from that function.
        """

        def warner() -> None:
            warnings.warn('first line warning')
            warnings.warn('internal line warning')
            warnings.warn('last line warning')
        warner()
        self.assertEqual(len(self.flushWarnings(offendingFunctions=[warner])), 3)

    def test_invalidFilter(self) -> None:
        """
        If an object which is neither a function nor a method is included in the
        C{offendingFunctions} list, C{flushWarnings} raises L{ValueError}.  Such
        a call flushes no warnings.
        """
        warnings.warn('oh no')
        self.assertRaises(ValueError, self.flushWarnings, [None])
        self.assertEqual(len(self.flushWarnings()), 1)

    def test_missingSource(self) -> None:
        """
        Warnings emitted by a function the source code of which is not
        available can still be flushed.
        """
        package = FilePath(self.mktemp().encode('utf-8')).child(b'twisted_private_helper')
        package.makedirs()
        package.child(b'__init__.py').setContent(b'')
        package.child(b'missingsourcefile.py').setContent(b'\nimport warnings\ndef foo():\n    warnings.warn("oh no")\n')
        pathEntry = package.parent().path.decode('utf-8')
        sys.path.insert(0, pathEntry)
        self.addCleanup(sys.path.remove, pathEntry)
        from twisted_private_helper import missingsourcefile
        self.addCleanup(sys.modules.pop, 'twisted_private_helper')
        self.addCleanup(sys.modules.pop, missingsourcefile.__name__)
        package.child(b'missingsourcefile.py').remove()
        missingsourcefile.foo()
        self.assertEqual(len(self.flushWarnings([missingsourcefile.foo])), 1)

    def test_renamedSource(self) -> None:
        """
        Warnings emitted by a function defined in a file which has been renamed
        since it was initially compiled can still be flushed.

        This is testing the code which specifically supports working around the
        unfortunate behavior of CPython to write a .py source file name into
        the .pyc files it generates and then trust that it is correct in
        various places.  If source files are renamed, .pyc files may not be
        regenerated, but they will contain incorrect filenames.
        """
        package = FilePath(self.mktemp().encode('utf-8')).child(b'twisted_private_helper')
        package.makedirs()
        package.child(b'__init__.py').setContent(b'')
        package.child(b'module.py').setContent(b'\nimport warnings\ndef foo():\n    warnings.warn("oh no")\n')
        pathEntry = package.parent().path.decode('utf-8')
        sys.path.insert(0, pathEntry)
        self.addCleanup(sys.path.remove, pathEntry)
        from twisted_private_helper import module
        del sys.modules['twisted_private_helper']
        del sys.modules[module.__name__]
        try:
            from importlib import invalidate_caches
        except ImportError:
            pass
        else:
            invalidate_caches()
        package.moveTo(package.sibling(b'twisted_renamed_helper'))
        from twisted_renamed_helper import module
        self.addCleanup(sys.modules.pop, 'twisted_renamed_helper')
        self.addCleanup(sys.modules.pop, module.__name__)
        module.foo()
        self.assertEqual(len(self.flushWarnings([module.foo])), 1)

    def test_offendingFunctions_deep_branch(self) -> None:
        """
        In Python 3.6 the dis.findlinestarts documented behaviour
        was changed such that the reported lines might not be sorted ascending.
        In Python 3.10 PEP 626 introduced byte-code change such that the last
        line of a function wasn't always associated with the last byte-code.
        In the past flushWarning was not detecting that such a function was
        associated with any warnings.
        """

        def foo(a: int=1, b: int=1) -> None:
            if a:
                if b:
                    warnings.warn('oh no')
                else:
                    pass
        foo()
        self.assertEqual(len(self.flushWarnings([foo])), 1)