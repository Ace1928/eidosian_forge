import os
import sys
from functools import partial
from os.path import isabs
from typing import (
from unittest import TestCase, TestSuite
from attrs import define, field, frozen
from attrs.converters import default_if_none
from twisted.internet.defer import Deferred, DeferredList, gatherResults
from twisted.internet.interfaces import IReactorCore, IReactorProcess
from twisted.logger import Logger
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.python.modules import theSystemPath
from .._asyncrunner import _iterateTests
from ..itrial import IReporter, ITestCase
from ..reporter import UncleanWarningsReporterWrapper
from ..runner import TestHolder
from ..util import _unusedTestDirectory, openTestLog
from . import _WORKER_AMP_STDIN, _WORKER_AMP_STDOUT
from .distreporter import DistReporter
from .functional import countingCalls, discardResult, iterateWhile, takeWhile
from .worker import LocalWorker, LocalWorkerAMP, WorkerAction
@define
class DistTrialRunner:
    """
    A specialized runner for distributed trial. The runner launches a number of
    local worker processes which will run tests.

    @ivar _maxWorkers: the number of workers to be spawned.

    @ivar _exitFirst: ``True`` to stop the run as soon as a test case fails.
        ``False`` to run through the whole suite and report all of the results
        at the end.

    @ivar stream: stream which the reporter will use.

    @ivar _reporterFactory: the reporter class to be used.
    """
    _distReporterFactory = DistReporter
    _logger = Logger()
    _reporterFactory: Callable[..., IReporter]
    _maxWorkers: int
    _workerArguments: List[str]
    _exitFirst: bool = False
    _reactor: IDistTrialReactor = field(default=None, converter=default_if_none(factory=_defaultReactor))
    stream: TextIO = field(default=None, converter=default_if_none(sys.stdout))
    _tracebackFormat: str = 'default'
    _realTimeErrors: bool = False
    _uncleanWarnings: bool = False
    _logfile: str = 'test.log'
    _workingDirectory: str = '_trial_temp'
    _workerPoolFactory: Callable[[WorkerPoolConfig], WorkerPool] = WorkerPool

    def _makeResult(self) -> DistReporter:
        """
        Make reporter factory, and wrap it with a L{DistReporter}.
        """
        reporter = self._reporterFactory(self.stream, self._tracebackFormat, realtime=self._realTimeErrors)
        if self._uncleanWarnings:
            reporter = UncleanWarningsReporterWrapper(reporter)
        return self._distReporterFactory(reporter)

    def writeResults(self, result):
        """
        Write test run final outcome to result.

        @param result: A C{TestResult} which will print errors and the summary.
        """
        result.done()

    async def _driveWorker(self, result: DistReporter, testCases: Sequence[ITestCase], worker: LocalWorkerAMP) -> None:
        """
        Drive a L{LocalWorkerAMP} instance, iterating the tests and calling
        C{run} for every one of them.

        @param worker: The L{LocalWorkerAMP} to drive.

        @param result: The global L{DistReporter} instance.

        @param testCases: The global list of tests to iterate.

        @return: A coroutine that completes after all of the tests have
            completed.
        """

        async def task(case):
            try:
                await worker.run(case, result)
            except Exception:
                result.original.addError(case, Failure())
        for case in testCases:
            await task(case)

    async def runAsync(self, suite: Union[TestCase, TestSuite], untilFailure: bool=False) -> DistReporter:
        """
        Spawn local worker processes and load tests. After that, run them.

        @param suite: A test or suite to be run.

        @param untilFailure: If C{True}, continue to run the tests until they
            fail.

        @return: A coroutine that completes with the test result.
        """
        testCases = list(_iterateTests(suite))
        poolStarter = self._workerPoolFactory(WorkerPoolConfig(min(len(testCases), self._maxWorkers), FilePath(self._workingDirectory), self._workerArguments, self._logfile))
        self.stream.write(f'Running {suite.countTestCases()} tests.\n')
        startedPool = await poolStarter.start(self._reactor)
        condition = partial(shouldContinue, untilFailure)

        @countingCalls
        async def runAndReport(n: int) -> DistReporter:
            if untilFailure:
                self.stream.write(f'Test Pass {n + 1}\n')
            result = self._makeResult()
            if self._exitFirst:
                casesCondition = lambda _: result.original.wasSuccessful()
            else:
                casesCondition = lambda _: True
            await runTests(startedPool, takeWhile(casesCondition, testCases), result, self._driveWorker)
            self.writeResults(result)
            return result
        try:
            return await iterateWhile(condition, runAndReport)
        finally:
            await startedPool.join()

    def _run(self, test: Union[TestCase, TestSuite], untilFailure: bool) -> IReporter:
        result: Union[Failure, DistReporter, None] = None
        reactorStopping: bool = False
        testsInProgress: Deferred[object]

        def capture(r: Union[Failure, DistReporter]) -> None:
            nonlocal result
            result = r

        def maybeStopTests() -> Optional[Deferred[object]]:
            nonlocal reactorStopping
            reactorStopping = True
            if result is None:
                testsInProgress.cancel()
                return testsInProgress
            return None

        def maybeStopReactor(result: object) -> object:
            if not reactorStopping:
                self._reactor.stop()
            return result
        self._reactor.addSystemEventTrigger('before', 'shutdown', maybeStopTests)
        testsInProgress = Deferred.fromCoroutine(self.runAsync(test, untilFailure)).addBoth(capture).addBoth(maybeStopReactor)
        self._reactor.run()
        if isinstance(result, Failure):
            result.raiseException()
        assert isinstance(result, DistReporter), f'{result} is not DistReporter'
        return cast(IReporter, result.original)

    def run(self, test: Union[TestCase, TestSuite]) -> IReporter:
        """
        Run a reactor and a test suite.

        @param test: The test or suite to run.
        """
        return self._run(test, untilFailure=False)

    def runUntilFailure(self, test: Union[TestCase, TestSuite]) -> IReporter:
        """
        Run the tests with local worker processes until they fail.

        @param test: The test or suite to run.
        """
        return self._run(test, untilFailure=True)