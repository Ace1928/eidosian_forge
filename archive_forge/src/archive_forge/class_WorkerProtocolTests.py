import os
from io import BytesIO, StringIO
from typing import Type
from unittest import TestCase as PyUnitTestCase
from zope.interface.verify import verifyObject
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost, ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist import managercommands
from twisted.trial._dist.worker import (
from twisted.trial.reporter import TestResult
from twisted.trial.test import pyunitcases, skipping
from twisted.trial.unittest import TestCase, makeTodo
from .matchers import isFailure, matches_result, similarFrame
class WorkerProtocolTests(TestCase):
    """
    Tests for L{WorkerProtocol}.
    """
    worker: WorkerProtocol
    server: LocalWorkerAMP

    def setUp(self) -> None:
        """
        Set up a transport, a result stream and a protocol instance.
        """
        self.worker, self.server, pump = connectedServerAndClient(LocalWorkerAMP, WorkerProtocol, greet=False)
        self.flush = pump.flush

    def test_run(self) -> None:
        """
        Sending the L{workercommands.Run} command to the worker returns a
        response with C{success} sets to C{True}.
        """
        d = Deferred.fromCoroutine(self.server.run(pyunitcases.PyUnitTest('test_pass'), TestResult()))
        self.flush()
        self.assertEqual({'success': True}, self.successResultOf(d))

    def test_start(self) -> None:
        """
        The C{start} command changes the current path.
        """
        curdir = os.path.realpath(os.path.curdir)
        self.addCleanup(os.chdir, curdir)
        self.worker.start('..')
        self.assertNotEqual(os.path.realpath(os.path.curdir), curdir)