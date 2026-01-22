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
class WorkerPoolTests(TestCase):
    """
    Tests for L{WorkerPool}.
    """

    def setUp(self):
        self.parent = FilePath(self.mktemp())
        self.workingDirectory = self.parent.child('_trial_temp')
        self.config = WorkerPoolConfig(numWorkers=4, workingDirectory=self.workingDirectory, workerArguments=[], logFile='out.log')
        self.pool = WorkerPool(self.config)

    def test_createLocalWorkers(self):
        """
        C{_createLocalWorkers} iterates the list of protocols and create one
        L{LocalWorker} for each.
        """
        protocols = [object() for x in range(4)]
        workers = self.pool._createLocalWorkers(protocols, FilePath('path'), StringIO())
        for s in workers:
            self.assertIsInstance(s, LocalWorker)
        self.assertEqual(4, len(workers))

    def test_launchWorkerProcesses(self):
        """
        Given a C{spawnProcess} function, C{_launchWorkerProcess} launches a
        python process with an existing path as its argument.
        """
        protocols = [ProcessProtocol() for i in range(4)]
        arguments = []
        environment = {}

        def fakeSpawnProcess(processProtocol, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
            arguments.append(executable)
            arguments.extend(args)
            environment.update(env)
        self.pool._launchWorkerProcesses(fakeSpawnProcess, protocols, ['foo'])
        self.assertEqual(arguments[0], arguments[1])
        self.assertTrue(os.path.exists(arguments[2]))
        self.assertEqual('foo', arguments[3])
        self.assertEqual(os.pathsep.join(sys.path), environment['PYTHONPATH'])

    def test_run(self):
        """
        C{run} dispatches the given action to each of its workers exactly once.
        """
        self.parent.makedirs()
        workers = []
        starting = self.pool.start(CountingReactor([]))
        started = self.successResultOf(starting)
        running = started.run(lambda w: succeed(workers.append(w)))
        self.successResultOf(running)
        assert_that(workers, has_length(self.config.numWorkers))

    def test_runUsedDirectory(self):
        """
        L{WorkerPool.start} checks if the test directory is already locked, and if
        it is generates a name based on it.
        """
        self.parent.makedirs()
        lock = FilesystemLock(self.workingDirectory.path + '.lock')
        self.assertTrue(lock.lock())
        self.addCleanup(lock.unlock)
        fakeReactor = CountingReactor([])
        started = self.successResultOf(self.pool.start(fakeReactor))
        self.assertEqual(started.workingDirectory, self.workingDirectory.sibling('_trial_temp-1'))

    def test_join(self):
        """
        L{StartedWorkerPool.join} causes all of the workers to exit, closes the
        log file, and unlocks the test directory.
        """
        self.parent.makedirs()
        reactor = CountingReactor([])
        started = self.successResultOf(self.pool.start(reactor))
        joining = Deferred.fromCoroutine(started.join())
        self.assertNoResult(joining)
        for w in reactor._workers:
            assert_that(w.transport._closed, contains(_WORKER_AMP_STDIN))
            for fd in w.transport._closed:
                w.childConnectionLost(fd)
            for f in [w.processExited, w.processEnded]:
                f(Failure(ProcessDone(0)))
        assert_that(self.successResultOf(joining), none())
        assert_that(started.testLog.closed, equal_to(True))
        assert_that(started.testDirLock.locked, equal_to(False))

    @given(booleans(), sampled_from(['out.log', f'subdir{sep}out.log']))
    def test_logFile(self, absolute: bool, logFile: str) -> None:
        """
        L{WorkerPool.start} creates a L{StartedWorkerPool} configured with a
        log file based on the L{WorkerPoolConfig.logFile}.
        """
        if absolute:
            logFile = self.parent.path + sep + logFile
        config = assoc(self.config, logFile=logFile)
        if absolute:
            matches = equal_to(logFile)
        else:
            matches = AllOf(starts_with(config.workingDirectory.path), ends_with(sep + logFile))
        pool = WorkerPool(config)
        started = self.successResultOf(pool.start(CountingReactor([])))
        assert_that(started.testLog.name, matches)