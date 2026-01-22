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
class FailureTests(SynchronousTestCase):
    """
    Tests for L{failure.Failure}.
    """

    def test_failAndTrap(self) -> None:
        """
        Trapping a L{Failure}.
        """
        try:
            raise NotImplementedError('test')
        except BaseException:
            f = failure.Failure()
        error = f.trap(SystemExit, RuntimeError)
        self.assertEqual(error, RuntimeError)
        self.assertEqual(f.type, NotImplementedError)

    def test_trapRaisesWrappedException(self) -> None:
        """
        If the wrapped C{Exception} is not a subclass of one of the
        expected types, L{failure.Failure.trap} raises the wrapped
        C{Exception}.
        """
        exception = ValueError()
        try:
            raise exception
        except BaseException:
            f = failure.Failure()
        untrapped = self.assertRaises(ValueError, f.trap, OverflowError)
        self.assertIs(exception, untrapped)

    def test_failureValueFromFailure(self) -> None:
        """
        A L{failure.Failure} constructed from another
        L{failure.Failure} instance, has its C{value} property set to
        the value of that L{failure.Failure} instance.
        """
        exception = ValueError()
        f1 = failure.Failure(exception)
        f2 = failure.Failure(f1)
        self.assertIs(f2.value, exception)

    def test_failureValueFromFoundFailure(self) -> None:
        """
        A L{failure.Failure} constructed without a C{exc_value}
        argument, will search for an "original" C{Failure}, and if
        found, its value will be used as the value for the new
        C{Failure}.
        """
        exception = ValueError()
        f1 = failure.Failure(exception)
        try:
            f1.trap(OverflowError)
        except BaseException:
            f2 = failure.Failure()
        self.assertIs(f2.value, exception)

    def assertStartsWith(self, s: str, prefix: str) -> None:
        """
        Assert that C{s} starts with a particular C{prefix}.

        @param s: The input string.
        @type s: C{str}
        @param prefix: The string that C{s} should start with.
        @type prefix: C{str}
        """
        self.assertTrue(s.startswith(prefix), f'{prefix!r} is not the start of {s!r}')

    def assertEndsWith(self, s: str, suffix: str) -> None:
        """
        Assert that C{s} end with a particular C{suffix}.

        @param s: The input string.
        @type s: C{str}
        @param suffix: The string that C{s} should end with.
        @type suffix: C{str}
        """
        self.assertTrue(s.endswith(suffix), f'{suffix!r} is not the end of {s!r}')

    def assertTracebackFormat(self, tb: str, prefix: str, suffix: str) -> None:
        """
        Assert that the C{tb} traceback contains a particular C{prefix} and
        C{suffix}.

        @param tb: The traceback string.
        @type tb: C{str}
        @param prefix: The string that C{tb} should start with.
        @type prefix: C{str}
        @param suffix: The string that C{tb} should end with.
        @type suffix: C{str}
        """
        self.assertStartsWith(tb, prefix)
        self.assertEndsWith(tb, suffix)

    def assertDetailedTraceback(self, captureVars: bool=False, cleanFailure: bool=False) -> None:
        """
        Assert that L{printDetailedTraceback} produces and prints a detailed
        traceback.

        The detailed traceback consists of a header::

          *--- Failure #20 ---

        The body contains the stacktrace::

          /twisted/trial/_synctest.py:1180: _run(...)
          /twisted/python/util.py:1076: runWithWarningsSuppressed(...)
          --- <exception caught here> ---
          /twisted/test/test_failure.py:39: getDivisionFailure(...)

        If C{captureVars} is enabled the body also includes a list of
        globals and locals::

           [ Locals ]
             exampleLocalVar : 'xyz'
             ...
           ( Globals )
             ...

        Or when C{captureVars} is disabled::

           [Capture of Locals and Globals disabled (use captureVars=True)]

        When C{cleanFailure} is enabled references to other objects are removed
        and replaced with strings.

        And finally the footer with the L{Failure}'s value::

          exceptions.ZeroDivisionError: float division
          *--- End of Failure #20 ---

        @param captureVars: Enables L{Failure.captureVars}.
        @type captureVars: C{bool}
        @param cleanFailure: Enables L{Failure.cleanFailure}.
        @type cleanFailure: C{bool}
        """
        if captureVars:
            exampleLocalVar = 'xyz'
            exampleLocalVar
        f = getDivisionFailure(captureVars=captureVars)
        out = StringIO()
        if cleanFailure:
            f.cleanFailure()
        f.printDetailedTraceback(out)
        tb = out.getvalue()
        start = '*--- Failure #%d%s---\n' % (f.count, f.pickled and ' (pickled) ' or ' ')
        end = '{}: {}\n*--- End of Failure #{} ---\n'.format(reflect.qual(f.type), reflect.safe_str(f.value), f.count)
        self.assertTracebackFormat(tb, start, end)
        linesWithVars = [line for line in tb.splitlines() if line.startswith('  ')]
        if captureVars:
            self.assertNotEqual([], linesWithVars)
            if cleanFailure:
                line = '  exampleLocalVar : "\'xyz\'"'
            else:
                line = "  exampleLocalVar : 'xyz'"
            self.assertIn(line, linesWithVars)
        else:
            self.assertEqual([], linesWithVars)
            self.assertIn(' [Capture of Locals and Globals disabled (use captureVars=True)]\n', tb)

    def assertBriefTraceback(self, captureVars: bool=False) -> None:
        """
        Assert that L{printBriefTraceback} produces and prints a brief
        traceback.

        The brief traceback consists of a header::

          Traceback: <type 'exceptions.ZeroDivisionError'>: float division

        The body with the stacktrace::

          /twisted/trial/_synctest.py:1180:_run
          /twisted/python/util.py:1076:runWithWarningsSuppressed

        And the footer::

          --- <exception caught here> ---
          /twisted/test/test_failure.py:39:getDivisionFailure

        @param captureVars: Enables L{Failure.captureVars}.
        @type captureVars: C{bool}
        """
        if captureVars:
            exampleLocalVar = 'abcde'
            exampleLocalVar
        f = getDivisionFailure()
        out = StringIO()
        f.printBriefTraceback(out)
        tb = out.getvalue()
        stack = ''
        for method, filename, lineno, localVars, globalVars in f.frames:
            stack += f'{filename}:{lineno}:{method}\n'
        zde = repr(ZeroDivisionError)
        self.assertTracebackFormat(tb, f'Traceback: {zde}: ', f'{failure.EXCEPTION_CAUGHT_HERE}\n{stack}')
        if captureVars:
            self.assertIsNone(re.search('exampleLocalVar.*abcde', tb))

    def assertDefaultTraceback(self, captureVars: bool=False) -> None:
        """
        Assert that L{printTraceback} produces and prints a default traceback.

        The default traceback consists of a header::

          Traceback (most recent call last):

        The body with traceback::

          File "/twisted/trial/_synctest.py", line 1180, in _run
             runWithWarningsSuppressed(suppress, method)

        And the footer::

          --- <exception caught here> ---
            File "twisted/test/test_failure.py", line 39, in getDivisionFailure
              1 / 0
            exceptions.ZeroDivisionError: float division

        @param captureVars: Enables L{Failure.captureVars}.
        @type captureVars: C{bool}
        """
        if captureVars:
            exampleLocalVar = 'xyzzy'
            exampleLocalVar
        f = getDivisionFailure(captureVars=captureVars)
        out = StringIO()
        f.printTraceback(out)
        tb = out.getvalue()
        stack = ''
        for method, filename, lineno, localVars, globalVars in f.frames:
            stack += f'  File "{filename}", line {lineno}, in {method}\n'
            stack += f'    {linecache.getline(filename, lineno).strip()}\n'
        self.assertTracebackFormat(tb, 'Traceback (most recent call last):', '%s\n%s%s: %s\n' % (failure.EXCEPTION_CAUGHT_HERE, stack, reflect.qual(f.type), reflect.safe_str(f.value)))
        if captureVars:
            self.assertIsNone(re.search('exampleLocalVar.*xyzzy', tb))

    def test_printDetailedTraceback(self) -> None:
        """
        L{printDetailedTraceback} returns a detailed traceback including the
        L{Failure}'s count.
        """
        self.assertDetailedTraceback()

    def test_printBriefTraceback(self) -> None:
        """
        L{printBriefTraceback} returns a brief traceback.
        """
        self.assertBriefTraceback()

    def test_printTraceback(self) -> None:
        """
        L{printTraceback} returns a traceback.
        """
        self.assertDefaultTraceback()

    def test_printDetailedTracebackCapturedVars(self) -> None:
        """
        L{printDetailedTraceback} captures the locals and globals for its
        stack frames and adds them to the traceback, when called on a
        L{Failure} constructed with C{captureVars=True}.
        """
        self.assertDetailedTraceback(captureVars=True)

    def test_printBriefTracebackCapturedVars(self) -> None:
        """
        L{printBriefTraceback} returns a brief traceback when called on a
        L{Failure} constructed with C{captureVars=True}.

        Local variables on the stack can not be seen in the resulting
        traceback.
        """
        self.assertBriefTraceback(captureVars=True)

    def test_printTracebackCapturedVars(self) -> None:
        """
        L{printTraceback} returns a traceback when called on a L{Failure}
        constructed with C{captureVars=True}.

        Local variables on the stack can not be seen in the resulting
        traceback.
        """
        self.assertDefaultTraceback(captureVars=True)

    def test_printDetailedTracebackCapturedVarsCleaned(self) -> None:
        """
        C{printDetailedTraceback} includes information about local variables on
        the stack after C{cleanFailure} has been called.
        """
        self.assertDetailedTraceback(captureVars=True, cleanFailure=True)

    def test_invalidFormatFramesDetail(self) -> None:
        """
        L{failure.format_frames} raises a L{ValueError} if the supplied
        C{detail} level is unknown.
        """
        self.assertRaises(ValueError, failure.format_frames, None, None, detail='noisia')

    def test_ExplictPass(self) -> None:
        e = RuntimeError()
        f = failure.Failure(e)
        f.trap(RuntimeError)
        self.assertEqual(f.value, e)

    def _getInnermostFrameLine(self, f: failure.Failure) -> str | None:
        try:
            f.raiseException()
        except ZeroDivisionError:
            tb = traceback.extract_tb(sys.exc_info()[2])
            return tb[-1].line
        else:
            raise Exception("f.raiseException() didn't raise ZeroDivisionError!?")

    def test_RaiseExceptionWithTB(self) -> None:
        f = getDivisionFailure()
        innerline = self._getInnermostFrameLine(f)
        self.assertEqual(innerline, '1 / 0')

    def test_stringExceptionConstruction(self) -> None:
        """
        Constructing a C{Failure} with a string as its exception value raises
        a C{TypeError}, as this is no longer supported as of Python 2.6.
        """
        exc = self.assertRaises(TypeError, failure.Failure, 'ono!')
        self.assertIn('Strings are not supported by Failure', str(exc))

    def test_ConstructionFails(self) -> None:
        """
        Creating a Failure with no arguments causes it to try to discover the
        current interpreter exception state.  If no such state exists, creating
        the Failure should raise a synchronous exception.
        """
        self.assertRaises(failure.NoCurrentExceptionError, failure.Failure)

    def test_getTracebackObject(self) -> None:
        """
        If the C{Failure} has not been cleaned, then C{getTracebackObject}
        returns the traceback object that captured in its constructor.
        """
        f = getDivisionFailure()
        self.assertEqual(f.getTracebackObject(), f.tb)

    def test_getTracebackObjectFromCaptureVars(self) -> None:
        """
        C{captureVars=True} has no effect on the result of
        C{getTracebackObject}.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            noVarsFailure = failure.Failure()
            varsFailure = failure.Failure(captureVars=True)
        self.assertEqual(noVarsFailure.getTracebackObject(), varsFailure.tb)

    def test_getTracebackObjectFromClean(self) -> None:
        """
        If the Failure has been cleaned, then C{getTracebackObject} returns an
        object that looks the same to L{traceback.extract_tb}.
        """
        f = getDivisionFailure()
        expected = traceback.extract_tb(f.getTracebackObject())
        f.cleanFailure()
        observed = traceback.extract_tb(f.getTracebackObject())
        self.assertIsNotNone(expected)
        self.assertEqual(expected, observed)

    def test_getTracebackObjectFromCaptureVarsAndClean(self) -> None:
        """
        If the Failure was created with captureVars, then C{getTracebackObject}
        returns an object that looks the same to L{traceback.extract_tb}.
        """
        f = getDivisionFailure(captureVars=True)
        expected = traceback.extract_tb(f.getTracebackObject())
        f.cleanFailure()
        observed = traceback.extract_tb(f.getTracebackObject())
        self.assertEqual(expected, observed)

    def test_getTracebackObjectWithoutTraceback(self) -> None:
        """
        L{failure.Failure}s need not be constructed with traceback objects. If
        a C{Failure} has no traceback information at all, C{getTracebackObject}
        just returns None.

        None is a good value, because traceback.extract_tb(None) -> [].
        """
        f = failure.Failure(Exception('some error'))
        self.assertIsNone(f.getTracebackObject())

    def test_tracebackFromExceptionInPython3(self) -> None:
        """
        If a L{failure.Failure} is constructed with an exception but no
        traceback in Python 3, the traceback will be extracted from the
        exception's C{__traceback__} attribute.
        """
        try:
            1 / 0
        except BaseException:
            klass, exception, tb = sys.exc_info()
        f = failure.Failure(exception)
        self.assertIs(f.tb, tb)

    def test_cleanFailureRemovesTracebackInPython3(self) -> None:
        """
        L{failure.Failure.cleanFailure} sets the C{__traceback__} attribute of
        the exception to L{None} in Python 3.
        """
        f = getDivisionFailure()
        self.assertIsNotNone(f.tb)
        self.assertIs(f.value.__traceback__, f.tb)
        f.cleanFailure()
        self.assertIsNone(f.value.__traceback__)

    def test_distb(self) -> None:
        """
        The traceback captured by a L{Failure} is compatible with the stdlib
        L{dis.distb} function as used in post-mortem debuggers. Specifically,
        it doesn't cause that function to raise an exception.
        """
        f = getDivisionFailure()
        buf = StringIO()
        distb(f.getTracebackObject(), file=buf)
        self.assertIn(' --> ', buf.getvalue())

    def test_repr(self) -> None:
        """
        The C{repr} of a L{failure.Failure} shows the type and string
        representation of the underlying exception.
        """
        f = getDivisionFailure()
        typeName = reflect.fullyQualifiedName(ZeroDivisionError)
        self.assertEqual(repr(f), '<twisted.python.failure.Failure %s: division by zero>' % (typeName,))