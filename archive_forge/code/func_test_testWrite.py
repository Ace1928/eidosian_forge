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
def test_testWrite(self) -> None:
    """
        L{LocalWorkerAMP.testWrite} writes the data received to its test
        stream.
        """
    stream = StringIO()
    self.managerAMP.setTestStream(stream)
    d = self.worker.callRemote(managercommands.TestWrite, out='Some output')
    self.flush()
    self.assertEqual({'success': True}, self.successResultOf(d))
    self.assertEqual('Some output\n', stream.getvalue())