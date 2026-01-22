import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
class StreamToExtendedDecorator(StreamResult):
    """Convert StreamResult API calls into ExtendedTestResult calls.

    This will buffer all calls for all concurrently active tests, and
    then flush each test as they complete.

    Incomplete tests will be flushed as errors when the test run stops.

    Non test file attachments are accumulated into a test called
    'testtools.extradata' flushed at the end of the run.
    """

    def __init__(self, decorated):
        self.decorated = ExtendedToOriginalDecorator(decorated)
        self.hook = _StreamToTestRecord(self._handle_tests)

    def status(self, test_id=None, test_status=None, *args, **kwargs):
        if test_status == 'exists':
            return
        self.hook.status(*args, test_id=test_id, test_status=test_status, **kwargs)

    def startTestRun(self):
        self.decorated.startTestRun()
        self.hook.startTestRun()

    def stopTestRun(self):
        self.hook.stopTestRun()
        self.decorated.stopTestRun()

    def _handle_tests(self, test_record):
        case = test_record.to_test_case()
        case.run(self.decorated)