import inspect
import os
import sys
import tempfile
import types
import unittest as pyunit
import warnings
from dis import findlinestarts as _findlinestarts
from typing import (
from unittest import SkipTest
from attrs import frozen
from typing_extensions import ParamSpec
from twisted.internet.defer import Deferred, ensureDeferred
from twisted.python import failure, log, monkey
from twisted.python.deprecate import (
from twisted.python.reflect import fullyQualifiedName
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import itrial, util
class SynchronousTestCase(_Assertions):
    """
    A unit test. The atom of the unit testing universe.

    This class extends C{unittest.TestCase} from the standard library.  A number
    of convenient testing helpers are added, including logging and warning
    integration, monkey-patching support, and more.

    To write a unit test, subclass C{SynchronousTestCase} and define a method
    (say, 'test_foo') on the subclass. To run the test, instantiate your
    subclass with the name of the method, and call L{run} on the instance,
    passing a L{TestResult} object.

    The C{trial} script will automatically find any C{SynchronousTestCase}
    subclasses defined in modules beginning with 'test_' and construct test
    cases for all methods beginning with 'test'.

    If an error is logged during the test run, the test will fail with an
    error. See L{log.err}.

    @ivar failureException: An exception class, defaulting to C{FailTest}. If
    the test method raises this exception, it will be reported as a failure,
    rather than an exception. All of the assertion methods raise this if the
    assertion fails.

    @ivar skip: L{None} or a string explaining why this test is to be
    skipped. If defined, the test will not be run. Instead, it will be
    reported to the result object as 'skipped' (if the C{TestResult} supports
    skipping).

    @ivar todo: L{None}, a string or a tuple of C{(errors, reason)} where
    C{errors} is either an exception class or an iterable of exception
    classes, and C{reason} is a string. See L{Todo} or L{makeTodo} for more
    information.

    @ivar suppress: L{None} or a list of tuples of C{(args, kwargs)} to be
    passed to C{warnings.filterwarnings}. Use these to suppress warnings
    raised in a test. Useful for testing deprecated code. See also
    L{util.suppress}.
    """
    failureException = FailTest

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self._passed = False
        self._cleanups = []
        self._testMethodName = methodName
        testMethod = getattr(self, methodName)
        self._parents = [testMethod, self, sys.modules.get(self.__class__.__module__)]

    def __eq__(self, other: object) -> bool:
        """
        Override the comparison defined by the base TestCase which considers
        instances of the same class with the same _testMethodName to be
        equal.  Since trial puts TestCase instances into a set, that
        definition of comparison makes it impossible to run the same test
        method twice.  Most likely, trial should stop using a set to hold
        tests, but until it does, this is necessary on Python 2.6. -exarkun
        """
        if isinstance(other, SynchronousTestCase):
            return self is other
        else:
            return NotImplemented

    def __hash__(self):
        return hash((self.__class__, self._testMethodName))

    def shortDescription(self):
        desc = super().shortDescription()
        if desc is None:
            return self._testMethodName
        return desc

    def getSkip(self) -> Tuple[bool, Optional[str]]:
        """
        Return the skip reason set on this test, if any is set. Checks on the
        instance first, then the class, then the module, then packages. As
        soon as it finds something with a C{skip} attribute, returns that in
        a tuple (L{True}, L{str}).
        If the C{skip} attribute does not exist, look for C{__unittest_skip__}
        and C{__unittest_skip_why__} attributes which are set by the standard
        library L{unittest.skip} function.
        Returns (L{False}, L{None}) if it cannot find anything.
        See L{TestCase} docstring for more details.
        """
        skipReason = util.acquireAttribute(self._parents, 'skip', None)
        doSkip = skipReason is not None
        if skipReason is None:
            doSkip = getattr(self, '__unittest_skip__', False)
            if doSkip:
                skipReason = getattr(self, '__unittest_skip_why__', '')
        return (doSkip, skipReason)

    def getTodo(self):
        """
        Return a L{Todo} object if the test is marked todo. Checks on the
        instance first, then the class, then the module, then packages. As
        soon as it finds something with a C{todo} attribute, returns that.
        Returns L{None} if it cannot find anything. See L{TestCase} docstring
        for more details.
        """
        todo = util.acquireAttribute(self._parents, 'todo', None)
        if todo is None:
            return None
        return makeTodo(todo)

    def runTest(self):
        """
        If no C{methodName} argument is passed to the constructor, L{run} will
        treat this method as the thing with the actual test inside.
        """

    def run(self, result):
        """
        Run the test case, storing the results in C{result}.

        First runs C{setUp} on self, then runs the test method (defined in the
        constructor), then runs C{tearDown}.  As with the standard library
        L{unittest.TestCase}, the return value of these methods is disregarded.
        In particular, returning a L{Deferred<twisted.internet.defer.Deferred>}
        has no special additional consequences.

        @param result: A L{TestResult} object.
        """
        log.msg('--> %s <--' % self.id())
        new_result = itrial.IReporter(result, None)
        if new_result is None:
            result = PyUnitResultAdapter(result)
        else:
            result = new_result
        result.startTest(self)
        doSkip, skipReason = self.getSkip()
        if doSkip:
            result.addSkip(self, skipReason)
            result.stopTest(self)
            return
        self._passed = False
        self._warnings = []
        self._installObserver()
        _collectWarnings(self._warnings.append, self._runFixturesAndTest, result)
        for w in self.flushWarnings():
            try:
                warnings.warn_explicit(**w)
            except BaseException:
                result.addError(self, failure.Failure())
        result.stopTest(self)

    def addCleanup(self, f: Callable[_P, object], *args: _P.args, **kwargs: _P.kwargs) -> None:
        """
        Add the given function to a list of functions to be called after the
        test has run, but before C{tearDown}.

        Functions will be run in reverse order of being added. This helps
        ensure that tear down complements set up.

        As with all aspects of L{SynchronousTestCase}, Deferreds are not
        supported in cleanup functions.
        """
        self._cleanups.append((f, args, kwargs))

    def patch(self, obj, attribute, value):
        """
        Monkey patch an object for the duration of the test.

        The monkey patch will be reverted at the end of the test using the
        L{addCleanup} mechanism.

        The L{monkey.MonkeyPatcher} is returned so that users can restore and
        re-apply the monkey patch within their tests.

        @param obj: The object to monkey patch.
        @param attribute: The name of the attribute to change.
        @param value: The value to set the attribute to.
        @return: A L{monkey.MonkeyPatcher} object.
        """
        monkeyPatch = monkey.MonkeyPatcher((obj, attribute, value))
        monkeyPatch.patch()
        self.addCleanup(monkeyPatch.restore)
        return monkeyPatch

    def flushLoggedErrors(self, *errorTypes):
        """
        Remove stored errors received from the log.

        C{TestCase} stores each error logged during the run of the test and
        reports them as errors during the cleanup phase (after C{tearDown}).

        @param errorTypes: If unspecified, flush all errors. Otherwise, only
        flush errors that match the given types.

        @return: A list of failures that have been removed.
        """
        return self._observer.flushErrors(*errorTypes)

    def flushWarnings(self, offendingFunctions=None):
        """
        Remove stored warnings from the list of captured warnings and return
        them.

        @param offendingFunctions: If L{None}, all warnings issued during the
            currently running test will be flushed.  Otherwise, only warnings
            which I{point} to a function included in this list will be flushed.
            All warnings include a filename and source line number; if these
            parts of a warning point to a source line which is part of a
            function, then the warning I{points} to that function.
        @type offendingFunctions: L{None} or L{list} of functions or methods.

        @raise ValueError: If C{offendingFunctions} is not L{None} and includes
            an object which is not a L{types.FunctionType} or
            L{types.MethodType} instance.

        @return: A C{list}, each element of which is a C{dict} giving
            information about one warning which was flushed by this call.  The
            keys of each C{dict} are:

                - C{'message'}: The string which was passed as the I{message}
                  parameter to L{warnings.warn}.

                - C{'category'}: The warning subclass which was passed as the
                  I{category} parameter to L{warnings.warn}.

                - C{'filename'}: The name of the file containing the definition
                  of the code object which was C{stacklevel} frames above the
                  call to L{warnings.warn}, where C{stacklevel} is the value of
                  the C{stacklevel} parameter passed to L{warnings.warn}.

                - C{'lineno'}: The source line associated with the active
                  instruction of the code object object which was C{stacklevel}
                  frames above the call to L{warnings.warn}, where
                  C{stacklevel} is the value of the C{stacklevel} parameter
                  passed to L{warnings.warn}.
        """
        if offendingFunctions is None:
            toFlush = self._warnings[:]
            self._warnings[:] = []
        else:
            toFlush = []
            for aWarning in self._warnings:
                for aFunction in offendingFunctions:
                    if not isinstance(aFunction, (types.FunctionType, types.MethodType)):
                        raise ValueError(f'{aFunction!r} is not a function or method')
                    aModule = sys.modules[aFunction.__module__]
                    filename = inspect.getabsfile(aModule)
                    if filename != os.path.normcase(aWarning.filename):
                        continue
                    lineNumbers = [lineNumber for _, lineNumber in _findlinestarts(aFunction.__code__) if lineNumber is not None]
                    if not min(lineNumbers) <= aWarning.lineno <= max(lineNumbers):
                        continue
                    toFlush.append(aWarning)
                    break
            list(map(self._warnings.remove, toFlush))
        return [{'message': w.message, 'category': w.category, 'filename': w.filename, 'lineno': w.lineno} for w in toFlush]

    def getDeprecatedModuleAttribute(self, moduleName, name, version, message=None):
        """
        Retrieve a module attribute which should have been deprecated,
        and assert that we saw the appropriate deprecation warning.

        @type moduleName: C{str}
        @param moduleName: Fully-qualified Python name of the module containing
            the deprecated attribute; if called from the same module as the
            attributes are being deprecated in, using the C{__name__} global can
            be helpful

        @type name: C{str}
        @param name: Attribute name which we expect to be deprecated

        @param version: The first L{version<twisted.python.versions.Version>} that
            the module attribute was deprecated.

        @type message: C{str}
        @param message: (optional) The expected deprecation message for the module attribute

        @return: The given attribute from the named module

        @raise FailTest: if no warnings were emitted on getattr, or if the
            L{DeprecationWarning} emitted did not produce the canonical
            please-use-something-else message that is standard for Twisted
            deprecations according to the given version and replacement.

        @since: Twisted 21.2.0
        """
        fqpn = moduleName + '.' + name
        module = sys.modules[moduleName]
        attr = getattr(module, name)
        warningsShown = self.flushWarnings([self.getDeprecatedModuleAttribute])
        if len(warningsShown) == 0:
            self.fail(f'{fqpn} is not deprecated.')
        observedWarning = warningsShown[0]['message']
        expectedWarning = DEPRECATION_WARNING_FORMAT % {'fqpn': fqpn, 'version': getVersionString(version)}
        if message is not None:
            expectedWarning = expectedWarning + ': ' + message
        self.assert_(observedWarning.startswith(expectedWarning), f'Expected {observedWarning!r} to start with {expectedWarning!r}')
        return attr

    def callDeprecated(self, version, f, *args, **kwargs):
        """
        Call a function that should have been deprecated at a specific version
        and in favor of a specific alternative, and assert that it was thusly
        deprecated.

        @param version: A 2-sequence of (since, replacement), where C{since} is
            a the first L{version<incremental.Version>} that C{f}
            should have been deprecated since, and C{replacement} is a suggested
            replacement for the deprecated functionality, as described by
            L{twisted.python.deprecate.deprecated}.  If there is no suggested
            replacement, this parameter may also be simply a
            L{version<incremental.Version>} by itself.

        @param f: The deprecated function to call.

        @param args: The arguments to pass to C{f}.

        @param kwargs: The keyword arguments to pass to C{f}.

        @return: Whatever C{f} returns.

        @raise Exception: Whatever C{f} raises.  If any exception is
            raised by C{f}, though, no assertions will be made about emitted
            deprecations.

        @raise FailTest: if no warnings were emitted by C{f}, or if the
            L{DeprecationWarning} emitted did not produce the canonical
            please-use-something-else message that is standard for Twisted
            deprecations according to the given version and replacement.
        """
        result = f(*args, **kwargs)
        warningsShown = self.flushWarnings([self.callDeprecated])
        try:
            info = list(version)
        except TypeError:
            since = version
            replacement = None
        else:
            [since, replacement] = info
        if len(warningsShown) == 0:
            self.fail(f'{f!r} is not deprecated.')
        observedWarning = warningsShown[0]['message']
        expectedWarning = getDeprecationWarningString(f, since, replacement=replacement)
        self.assertEqual(expectedWarning, observedWarning)
        return result

    def mktemp(self):
        """
        Create a new path name which can be used for a new file or directory.

        The result is a relative path that is guaranteed to be unique within the
        current working directory.  The parent of the path will exist, but the
        path will not.

        For a temporary directory call os.mkdir on the path.  For a temporary
        file just create the file (e.g. by opening the path for writing and then
        closing it).

        @return: The newly created path
        @rtype: C{str}
        """
        MAX_FILENAME = 32
        base = os.path.join(self.__class__.__module__[:MAX_FILENAME], self.__class__.__name__[:MAX_FILENAME], self._testMethodName[:MAX_FILENAME])
        if not os.path.exists(base):
            os.makedirs(base)
        dirname = os.path.relpath(tempfile.mkdtemp('', '', base))
        return os.path.join(dirname, 'temp')

    def _getSuppress(self):
        """
        Returns any warning suppressions set for this test. Checks on the
        instance first, then the class, then the module, then packages. As
        soon as it finds something with a C{suppress} attribute, returns that.
        Returns any empty list (i.e. suppress no warnings) if it cannot find
        anything. See L{TestCase} docstring for more details.
        """
        return util.acquireAttribute(self._parents, 'suppress', [])

    def _getSkipReason(self, method, skip):
        """
        Return the reason to use for skipping a test method.

        @param method: The method which produced the skip.
        @param skip: A L{unittest.SkipTest} instance raised by C{method}.
        """
        if len(skip.args) > 0:
            return skip.args[0]
        warnAboutFunction(method, 'Do not raise unittest.SkipTest with no arguments! Give a reason for skipping tests!')
        return skip

    def _run(self, suppress, todo, method, result):
        """
        Run a single method, either a test method or fixture.

        @param suppress: Any warnings to suppress, as defined by the C{suppress}
            attribute on this method, test case, or the module it is defined in.

        @param todo: Any expected failure or failures, as defined by the C{todo}
            attribute on this method, test case, or the module it is defined in.

        @param method: The method to run.

        @param result: The TestResult instance to which to report results.

        @return: C{True} if the method fails and no further method/fixture calls
            should be made, C{False} otherwise.
        """
        if inspect.isgeneratorfunction(method):
            exc = TypeError('{!r} is a generator function and therefore will never run'.format(method))
            result.addError(self, failure.Failure(exc))
            return True
        try:
            runWithWarningsSuppressed(suppress, method)
        except SkipTest as e:
            result.addSkip(self, self._getSkipReason(method, e))
        except BaseException:
            reason = failure.Failure()
            if todo is None or not todo.expected(reason):
                if reason.check(self.failureException):
                    addResult = result.addFailure
                else:
                    addResult = result.addError
                addResult(self, reason)
            else:
                result.addExpectedFailure(self, reason, todo)
        else:
            return False
        return True

    def _runFixturesAndTest(self, result):
        """
        Run C{setUp}, a test method, test cleanups, and C{tearDown}.

        @param result: The TestResult instance to which to report results.
        """
        suppress = self._getSuppress()
        try:
            if self._run(suppress, None, self.setUp, result):
                return
            todo = self.getTodo()
            method = getattr(self, self._testMethodName)
            failed = self._run(suppress, todo, method, result)
        finally:
            self._runCleanups(result)
        if todo and (not failed):
            result.addUnexpectedSuccess(self, todo)
        if self._run(suppress, None, self.tearDown, result):
            failed = True
        for error in self._observer.getErrors():
            result.addError(self, error)
            failed = True
        self._observer.flushErrors()
        self._removeObserver()
        if not (failed or todo):
            result.addSuccess(self)

    def _runCleanups(self, result):
        """
        Synchronously run any cleanups which have been added.
        """
        while len(self._cleanups) > 0:
            f, args, kwargs = self._cleanups.pop()
            try:
                f(*args, **kwargs)
            except BaseException:
                f = failure.Failure()
                result.addError(self, f)

    def _installObserver(self):
        self._observer = _logObserver
        self._observer._add()

    def _removeObserver(self):
        self._observer._remove()