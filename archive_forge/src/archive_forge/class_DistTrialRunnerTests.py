import os
import sys
from functools import partial
from io import StringIO
from os.path import sep
from typing import Callable, List, Set
from unittest import TestCase as PyUnitTestCase
from zope.interface import implementer, verify
from attrs import Factory, assoc, define, field
from hamcrest import (
from hamcrest.core.core.allof import AllOf
from hypothesis import given
from hypothesis.strategies import booleans, sampled_from
from twisted.internet import interfaces
from twisted.internet.base import ReactorBase
from twisted.internet.defer import CancelledError, Deferred, succeed
from twisted.internet.error import ProcessDone
from twisted.internet.protocol import ProcessProtocol, Protocol
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.internet.testing import MemoryReactorClock
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.lockfile import FilesystemLock
from twisted.trial._dist import _WORKER_AMP_STDIN
from twisted.trial._dist.distreporter import DistReporter
from twisted.trial._dist.disttrial import DistTrialRunner, WorkerPool, WorkerPoolConfig
from twisted.trial._dist.functional import (
from twisted.trial._dist.worker import LocalWorker, RunResult, Worker, WorkerAction
from twisted.trial.reporter import (
from twisted.trial.runner import ErrorHolder, TrialSuite
from twisted.trial.unittest import SynchronousTestCase, TestCase
from ...test import erroneous, sample
from .matchers import matches_result
class DistTrialRunnerTests(TestCase):
    """
    Tests for L{DistTrialRunner}.
    """
    suite = TrialSuite([sample.FooTest('test_foo')])

    def getRunner(self, **overrides):
        """
        Create a runner for testing.
        """
        args = dict(reporterFactory=TreeReporter, workingDirectory=self.mktemp(), stream=StringIO(), maxWorkers=4, workerArguments=[], workerPoolFactory=partial(LocalWorkerPool, autostop=True), reactor=CountingReactor([]))
        args.update(overrides)
        return DistTrialRunner(**args)

    def test_writeResults(self):
        """
        L{DistTrialRunner.writeResults} writes to the stream specified in the
        init.
        """
        stringIO = StringIO()
        result = DistReporter(Reporter(stringIO))
        runner = self.getRunner()
        runner.writeResults(result)
        self.assertTrue(stringIO.tell() > 0)

    def test_minimalWorker(self):
        """
        L{DistTrialRunner.runAsync} doesn't try to start more workers than the
        number of tests.
        """
        pool = None

        def recordingFactory(*a, **kw):
            nonlocal pool
            pool = LocalWorkerPool(*a, autostop=True, **kw)
            return pool
        maxWorkers = 7
        numTests = 3
        runner = self.getRunner(maxWorkers=maxWorkers, workerPoolFactory=recordingFactory)
        suite = TrialSuite([TestCase() for n in range(numTests)])
        self.successResultOf(runner.runAsync(suite))
        assert_that(pool._started[0].workers, has_length(numTests))

    def test_runUncleanWarnings(self) -> None:
        """
        Running with the C{unclean-warnings} option makes L{DistTrialRunner} uses
        the L{UncleanWarningsReporterWrapper}.
        """
        runner = self.getRunner(uncleanWarnings=True)
        d = runner.runAsync(self.suite)
        result = self.successResultOf(d)
        self.assertIsInstance(result, DistReporter)
        self.assertIsInstance(result.original, UncleanWarningsReporterWrapper)

    def test_runWithoutTest(self):
        """
        L{DistTrialRunner} can run an empty test suite.
        """
        stream = StringIO()
        runner = self.getRunner(stream=stream)
        result = self.successResultOf(runner.runAsync(TrialSuite()))
        self.assertIsInstance(result, DistReporter)
        output = stream.getvalue()
        self.assertIn('Running 0 test', output)
        self.assertIn('PASSED', output)

    def test_runWithoutTestButWithAnError(self):
        """
        Even if there is no test, the suite can contain an error (most likely,
        an import error): this should make the run fail, and the error should
        be printed.
        """
        err = ErrorHolder('an error', Failure(RuntimeError('foo bar')))
        stream = StringIO()
        runner = self.getRunner(stream=stream)
        result = self.successResultOf(runner.runAsync(err))
        self.assertIsInstance(result, DistReporter)
        output = stream.getvalue()
        self.assertIn('Running 0 test', output)
        self.assertIn('foo bar', output)
        self.assertIn('an error', output)
        self.assertIn('errors=1', output)
        self.assertIn('FAILED', output)

    def test_runUnexpectedError(self) -> None:
        """
        If for some reasons we can't connect to the worker process, the error is
        recorded in the result object.
        """
        runner = self.getRunner(workerPoolFactory=BrokenWorkerPool)
        result = self.successResultOf(runner.runAsync(self.suite))
        errors = result.original.errors
        assert_that(errors, has_length(1))
        assert_that(errors[0][1].type, equal_to(WorkerPoolBroken))

    def test_runUnexpectedErrorCtrlC(self) -> None:
        """
        If the reactor is stopped by C-c (i.e. `run` returns before the test
        case's Deferred has been fired) we should cancel the pending test run.
        """
        runner = self.getRunner(workerPoolFactory=LocalWorkerPool)
        with self.assertRaises(CancelledError):
            runner.run(self.suite)

    def test_runUnexpectedWorkerError(self) -> None:
        """
        If for some reason the worker process cannot run a test, the error is
        recorded in the result object.
        """
        runner = self.getRunner(workerPoolFactory=partial(LocalWorkerPool, workerFactory=_BrokenLocalWorker, autostop=True))
        result = self.successResultOf(runner.runAsync(self.suite))
        errors = result.original.errors
        assert_that(errors, has_length(1))
        assert_that(errors[0][1].type, equal_to(WorkerBroken))

    def test_runWaitForProcessesDeferreds(self) -> None:
        """
        L{DistTrialRunner} waits for the worker pool to stop.
        """
        pool = None

        def recordingFactory(*a, **kw):
            nonlocal pool
            pool = LocalWorkerPool(*a, autostop=False, **kw)
            return pool
        runner = self.getRunner(workerPoolFactory=recordingFactory)
        d = Deferred.fromCoroutine(runner.runAsync(self.suite))
        if pool is None:
            self.fail('worker pool was never created')
        assert pool is not None
        stopped = pool._started[0]._stopped
        self.assertNoResult(d)
        stopped.callback(None)
        result = self.successResultOf(d)
        self.assertIsInstance(result, DistReporter)

    def test_exitFirst(self):
        """
        L{DistTrialRunner} can run in C{exitFirst} mode where it will run until a
        test fails and then abandon the rest of the suite.
        """
        stream = StringIO()
        suite = TrialSuite([sample.FooTest('test_foo'), erroneous.TestRegularFail('test_fail'), sample.FooTest('test_bar')])
        runner = self.getRunner(stream=stream, exitFirst=True, maxWorkers=2)
        d = runner.runAsync(suite)
        result = self.successResultOf(d)
        assert_that(result.original, matches_result(successes=1, failures=has_length(1)))

    def test_runUntilFailure(self):
        """
        L{DistTrialRunner} can run in C{untilFailure} mode where it will run
        the given tests until they fail.
        """
        stream = StringIO()
        case = erroneous.EventuallyFailingTestCase('test_it')
        runner = self.getRunner(stream=stream)
        d = runner.runAsync(case, untilFailure=True)
        result = self.successResultOf(d)
        self.assertEqual(5, case.n)
        self.assertFalse(result.wasSuccessful())
        output = stream.getvalue()
        self.assertEqual(output.count('PASSED'), case.n - 1, 'expected to see PASSED in output')
        self.assertIn('FAIL', output)
        for i in range(1, 6):
            self.assertIn(f'Test Pass {i}', output)
        self.assertEqual(output.count('Ran 1 tests in'), case.n, 'expected to see per-iteration test count in output')

    def test_run(self) -> None:
        """
        L{DistTrialRunner.run} returns a L{DistReporter} containing the result of
        the test suite run.
        """
        runner = self.getRunner()
        result = runner.run(self.suite)
        assert_that(result.wasSuccessful(), equal_to(True))
        assert_that(result.successes, equal_to(1))

    def test_installedReactor(self) -> None:
        """
        L{DistTrialRunner.run} uses the installed reactor L{DistTrialRunner} was
        constructed without a reactor.
        """
        reactor = CountingReactor([])
        with AlternateReactor(reactor):
            runner = self.getRunner(reactor=None)
        result = runner.run(self.suite)
        assert_that(result.errors, equal_to([]))
        assert_that(result.failures, equal_to([]))
        assert_that(result.wasSuccessful(), equal_to(True))
        assert_that(result.successes, equal_to(1))
        assert_that(reactor.runCount, equal_to(1))
        assert_that(reactor.stopCount, equal_to(1))

    def test_wrongInstalledReactor(self) -> None:
        """
        L{DistTrialRunner} raises L{TypeError} if the installed reactor provides
        neither L{IReactorCore} nor L{IReactorProcess} and no other reactor is
        given.
        """

        class Core(ReactorBase):

            def installWaker(self):
                pass

        @implementer(interfaces.IReactorProcess)
        class Process:

            def spawnProcess(self, processProtocol, executable, args, env=None, path=None, uid=None, gid=None, usePTY=False, childFDs=None):
                pass

        class Neither:
            pass
        with AlternateReactor(Neither()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)
        with AlternateReactor(Core()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)
        with AlternateReactor(Process()):
            with self.assertRaises(TypeError):
                self.getRunner(reactor=None)

    def test_runFailure(self):
        """
        If there is an unexpected exception running the test suite then it is
        re-raised by L{DistTrialRunner.run}.
        """

        class BrokenFactory(Exception):
            pass

        def brokenFactory(*args, **kwargs):
            raise BrokenFactory()
        runner = self.getRunner(workerPoolFactory=brokenFactory)
        with self.assertRaises(BrokenFactory):
            runner.run(self.suite)