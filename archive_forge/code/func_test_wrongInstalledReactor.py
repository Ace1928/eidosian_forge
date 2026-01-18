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