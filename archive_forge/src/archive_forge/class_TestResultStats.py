import os
import re
import subprocess
import sys
import unittest
from io import BytesIO
from io import UnsupportedOperation as _UnsupportedOperation
import iso8601
from testtools import ExtendedToOriginalDecorator, content, content_type
from testtools.compat import _b, _u
from testtools.content import TracebackContent
from testtools import CopyStreamResult, testresult
from subunit import chunked, details
from subunit.v2 import ByteStreamToStreamResult, StreamResultToBytes
class TestResultStats(testresult.TestResult):
    """A pyunit TestResult interface implementation for making statistics.

    :ivar total_tests: The total tests seen.
    :ivar passed_tests: The tests that passed.
    :ivar failed_tests: The tests that failed.
    :ivar seen_tags: The tags seen across all tests.
    """

    def __init__(self, stream):
        """Create a TestResultStats which outputs to stream."""
        testresult.TestResult.__init__(self)
        self._stream = stream
        self.failed_tests = 0
        self.skipped_tests = 0
        self.seen_tags = set()

    @property
    def total_tests(self):
        return self.testsRun

    def addError(self, test, err, details=None):
        self.failed_tests += 1

    def addFailure(self, test, err, details=None):
        self.failed_tests += 1

    def addSkip(self, test, reason, details=None):
        self.skipped_tests += 1

    def formatStats(self):
        self._stream.write('Total tests:   %5d\n' % self.total_tests)
        self._stream.write('Passed tests:  %5d\n' % self.passed_tests)
        self._stream.write('Failed tests:  %5d\n' % self.failed_tests)
        self._stream.write('Skipped tests: %5d\n' % self.skipped_tests)
        tags = sorted(self.seen_tags)
        self._stream.write('Seen tags: %s\n' % ', '.join(tags))

    @property
    def passed_tests(self):
        return self.total_tests - self.failed_tests - self.skipped_tests

    def tags(self, new_tags, gone_tags):
        """Accumulate the seen tags."""
        self.seen_tags.update(new_tags)

    def wasSuccessful(self):
        """Tells whether or not this result was a success"""
        return self.failed_tests == 0