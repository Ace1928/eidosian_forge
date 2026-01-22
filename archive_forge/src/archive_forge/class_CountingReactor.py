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
@implementer(interfaces.IReactorProcess)
class CountingReactor(MemoryReactorClock):
    """
    A fake reactor that counts the calls to L{IReactorCore.run},
    L{IReactorCore.stop}, and L{IReactorProcess.spawnProcess}.
    """
    spawnCount = 0
    stopCount = 0
    runCount = 0

    def __init__(self, workers):
        MemoryReactorClock.__init__(self)
        self._workers = workers

    def spawnProcess(self, workerProto, executable, args=(), env={}, path=None, uid=None, gid=None, usePTY=0, childFDs=None):
        """
        See L{IReactorProcess.spawnProcess}.

        @param workerProto: See L{IReactorProcess.spawnProcess}.
        @param args: See L{IReactorProcess.spawnProcess}.
        @param kwargs: See L{IReactorProcess.spawnProcess}.
        """
        self._workers.append(workerProto)
        workerProto.makeConnection(FakeTransport())
        self.spawnCount += 1

    def stop(self):
        """
        See L{IReactorCore.stop}.
        """
        MemoryReactorClock.stop(self)
        if 'before' in self.triggers:
            self.triggers['before']['shutdown'][0][0]()
        self.stopCount += 1

    def run(self):
        """
        See L{IReactorCore.run}.
        """
        self.runCount += 1
        self.running = True
        self.hasRun = True
        for f, args, kwargs in self.whenRunningHooks:
            f(*args, **kwargs)
        self.stop()
        self.stopCount -= 1