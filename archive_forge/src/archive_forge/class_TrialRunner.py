import doctest
import importlib
import inspect
import os
import sys
import types
import unittest as pyunit
import warnings
from contextlib import contextmanager
from importlib.machinery import SourceFileLoader
from typing import Callable, Generator, List, Optional, TextIO, Type, Union
from zope.interface import implementer
from attrs import define
from typing_extensions import ParamSpec, Protocol, TypeAlias, TypeGuard
from twisted.internet import defer
from twisted.python import failure, filepath, log, modules, reflect
from twisted.trial import unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator, _iterateTests
from twisted.trial._synctest import _logObserver
from twisted.trial.itrial import ITestCase
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.unittest import TestSuite
from . import itrial
@define
class TrialRunner:
    """
    A specialised runner that the trial front end uses.

    @ivar reporterFactory: A callable to create a reporter to use.

    @ivar mode: Either C{None} for a normal test run, L{TrialRunner.DEBUG} for
        a run in the debugger, or L{TrialRunner.DRY_RUN} to collect and report
        the tests but not call any of them.

    @ivar logfile: The path to the file to write the test run log.

    @ivar stream: The file to report results to.

    @ivar profile: C{True} to run the tests with a profiler enabled.

    @ivar _tracebackFormat: A format name to use with L{Failure} for reporting
        failures.

    @ivar _realTimeErrors: C{True} if errors should be reported as they
        happen.  C{False} if they should only be reported at the end of the
        test run in the summary.

    @ivar uncleanWarnings: C{True} to report dirty reactor errors as warnings,
        C{False} to report them as test-failing errors.

    @ivar workingDirectory: A path template to a directory which will be the
        process's working directory while the tests are running.

    @ivar _forceGarbageCollection: C{True} to perform a full garbage
        collection at least after each test.  C{False} to let garbage
        collection run only when it normally would.

    @ivar debugger: In debug mode, an object to use to launch the debugger.

    @ivar _exitFirst: C{True} to stop after the first failed test.  C{False}
        to run the whole suite.

    @ivar log: An object to give to the reporter to use as a log publisher.
    """
    DEBUG = 'debug'
    DRY_RUN = 'dry-run'
    reporterFactory: Callable[[TextIO, str, bool, log.LogPublisher], itrial.IReporter]
    mode: Optional[str] = None
    logfile: str = 'test.log'
    stream: TextIO = sys.stdout
    profile: bool = False
    _tracebackFormat: str = 'default'
    _realTimeErrors: bool = False
    uncleanWarnings: bool = False
    workingDirectory: str = '_trial_temp'
    _forceGarbageCollection: bool = False
    debugger: Optional[_Debugger] = None
    _exitFirst: bool = False
    _log: log.LogPublisher = log

    def _makeResult(self) -> itrial.IReporter:
        reporter = self.reporterFactory(self.stream, self.tbformat, self.rterrors, self._log)
        if self._exitFirst:
            reporter = _ExitWrapper(reporter)
        if self.uncleanWarnings:
            reporter = UncleanWarningsReporterWrapper(reporter)
        return reporter

    @property
    def tbformat(self) -> str:
        return self._tracebackFormat

    @property
    def rterrors(self) -> bool:
        return self._realTimeErrors

    def run(self, test: Union[pyunit.TestCase, pyunit.TestSuite]) -> itrial.IReporter:
        """
        Run the test or suite and return a result object.
        """
        test = unittest.decorate(test, ITestCase)
        if self.profile:
            run = util.profiled(self._runWithoutDecoration, 'profile.data')
        else:
            run = self._runWithoutDecoration
        return run(test, self._forceGarbageCollection)

    def _runWithoutDecoration(self, test: Union[pyunit.TestCase, pyunit.TestSuite], forceGarbageCollection: bool=False) -> itrial.IReporter:
        """
        Private helper that runs the given test but doesn't decorate it.
        """
        result = self._makeResult()
        suite = TrialSuite([test], forceGarbageCollection)
        if self.mode == self.DRY_RUN:
            for single in _iterateTests(suite):
                result.startTest(single)
                result.addSuccess(single)
                result.stopTest(single)
        else:
            if self.mode == self.DEBUG:
                assert self.debugger is not None
                run = lambda: self.debugger.runcall(suite.run, result)
            else:
                run = lambda: suite.run(result)
            with _testDirectory(self.workingDirectory), _logFile(self.logfile):
                run()
        result.done()
        return result

    def runUntilFailure(self, test: Union[pyunit.TestCase, pyunit.TestSuite]) -> itrial.IReporter:
        """
        Repeatedly run C{test} until it fails.
        """
        count = 0
        while True:
            count += 1
            self.stream.write('Test Pass %d\n' % (count,))
            if count == 1:
                result = self.run(test)
            else:
                result = self._runWithoutDecoration(test)
            if result.testsRun == 0:
                break
            if not result.wasSuccessful():
                break
        return result