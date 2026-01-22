from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
class ExtendedGeneratorTests(SynchronousTestCase):
    """
    Tests C{failure.Failure} support for generator features added in Python 2.5
    """

    def _throwIntoGenerator(self, f: failure.Failure, g: Generator[Any, Any, Any]) -> None:
        try:
            f.throwExceptionIntoGenerator(g)
        except StopIteration:
            pass
        else:
            self.fail('throwExceptionIntoGenerator should have raised StopIteration')

    def test_throwExceptionIntoGenerator(self) -> None:
        """
        It should be possible to throw the exception that a Failure
        represents into a generator.
        """
        stuff = []

        def generator() -> Generator[None, None, None]:
            try:
                yield
            except BaseException:
                stuff.append(sys.exc_info())
            else:
                self.fail('Yield should have yielded exception.')
        g = generator()
        f = getDivisionFailure()
        next(g)
        self._throwIntoGenerator(f, g)
        self.assertEqual(stuff[0][0], ZeroDivisionError)
        self.assertIsInstance(stuff[0][1], ZeroDivisionError)
        self.assertEqual(traceback.extract_tb(stuff[0][2])[-1][-1], '1 / 0')

    def test_findFailureInGenerator(self) -> None:
        """
        Within an exception handler, it should be possible to find the
        original Failure that caused the current exception (if it was
        caused by throwExceptionIntoGenerator).
        """
        f = getDivisionFailure()
        f.cleanFailure()
        foundFailures = []

        def generator() -> Generator[None, None, None]:
            try:
                yield
            except BaseException:
                foundFailures.append(failure.Failure._findFailure())
            else:
                self.fail('No exception sent to generator')
        g = generator()
        next(g)
        self._throwIntoGenerator(f, g)
        self.assertEqual(foundFailures, [f])

    def test_failureConstructionFindsOriginalFailure(self) -> None:
        """
        When a Failure is constructed in the context of an exception
        handler that is handling an exception raised by
        throwExceptionIntoGenerator, the new Failure should be chained to that
        original Failure.
        """
        f = getDivisionFailure()
        f.cleanFailure()
        original_failure_str = f.getTraceback()
        newFailures = []

        def generator() -> Generator[None, None, None]:
            try:
                yield
            except BaseException:
                newFailures.append(failure.Failure())
            else:
                self.fail('No exception sent to generator')
        g = generator()
        next(g)
        self._throwIntoGenerator(f, g)
        self.assertEqual(len(newFailures), 1)
        self.assertEqual(original_failure_str, f.getTraceback())
        self.assertNotEqual(newFailures[0].getTraceback(), f.getTraceback())
        self.assertIn('generator', newFailures[0].getTraceback())
        self.assertNotIn('generator', f.getTraceback())

    def test_ambiguousFailureInGenerator(self) -> None:
        """
        When a generator reraises a different exception,
        L{Failure._findFailure} inside the generator should find the reraised
        exception rather than original one.
        """

        def generator() -> Generator[None, None, None]:
            try:
                try:
                    yield
                except BaseException:
                    [][1]
            except BaseException:
                self.assertIsInstance(failure.Failure().value, IndexError)
        g = generator()
        next(g)
        f = getDivisionFailure()
        self._throwIntoGenerator(f, g)

    def test_ambiguousFailureFromGenerator(self) -> None:
        """
        When a generator reraises a different exception,
        L{Failure._findFailure} above the generator should find the reraised
        exception rather than original one.
        """

        def generator() -> Generator[None, None, None]:
            try:
                yield
            except BaseException:
                [][1]
        g = generator()
        next(g)
        f = getDivisionFailure()
        try:
            self._throwIntoGenerator(f, g)
        except BaseException:
            self.assertIsInstance(failure.Failure().value, IndexError)