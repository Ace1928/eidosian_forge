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
def test_processEnded(self):
    """
        L{LocalWorker.processEnded} calls C{connectionLost} on itself and on
        the L{AMP} protocol.
        """
    transport = FakeTransport()
    protocol = SpyDataLocalWorkerAMP()
    localWorker = LocalWorker(protocol, FilePath(self.mktemp()), 'test.log')
    localWorker.makeConnection(transport)
    localWorker.processEnded(Failure(ProcessDone(0)))
    self.assertTrue(localWorker._outLog.closed)
    self.assertTrue(localWorker._errLog.closed)
    self.assertIdentical(None, protocol.transport)
    return self.assertFailure(localWorker.endDeferred, ProcessDone)