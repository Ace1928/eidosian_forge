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
class StartedWorkerPool:
    """
    A pool of workers which have already been started.

    @ivar workingDirectory: A directory holding the working directories for
        each of the workers.

    @ivar testDirLock: An object representing the cooperative lock this pool
        holds on its working directory.

    @ivar testLog: The open overall test log file.

    @ivar workers: Objects corresponding to the worker child processes and
        adapting between process-related interfaces and C{IProtocol}.

    @ivar ampWorkers: AMP protocol instances corresponding to the worker child
        processes.
    """
    workingDirectory: FilePath[Any]
    testDirLock: FilesystemLock
    testLog: TextIO
    workers: List[LocalWorker]
    ampWorkers: List[LocalWorkerAMP]
    _logger = Logger()

    async def run(self, workerAction: WorkerAction[Any]) -> None:
        """
        Run an action on all of the workers in the pool.
        """
        await gatherResults((discardResult(workerAction(worker)) for worker in self.ampWorkers))
        return None

    async def join(self) -> None:
        """
        Shut down all of the workers in the pool.

        The pool is unusable after this method is called.
        """
        results = await DeferredList([Deferred.fromCoroutine(worker.exit()) for worker in self.workers], consumeErrors=True)
        for n, (succeeded, failure) in enumerate(results):
            if not succeeded:
                self._logger.failure(f'joining disttrial worker #{n} failed', failure)
        del self.workers[:]
        del self.ampWorkers[:]
        self.testLog.close()
        self.testDirLock.unlock()