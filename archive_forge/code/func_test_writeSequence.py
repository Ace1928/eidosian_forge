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
def test_writeSequence(self):
    """
        L{LocalWorkerTransport.writeSequence} forwards the written data to the
        given transport.
        """
    transport = FakeTransport()
    localTransport = LocalWorkerTransport(transport)
    data = (b'The quick ', b'brown fox jumps ', b'over the lazy dog')
    localTransport.writeSequence(data)
    self.assertEqual(b''.join(data), transport.dataString)