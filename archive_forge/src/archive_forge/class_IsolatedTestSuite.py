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
class IsolatedTestSuite(unittest.TestSuite):
    """A TestSuite which runs its tests in a forked process.

    This decorator that will fork() before running the tests and report the
    results from the child process using a Subunit stream.  This is useful for
    handling tests that mutate global state, or are testing C extensions that
    could crash the VM.
    """

    def run(self, result=None):
        if result is None:
            result = testresult.TestResult()
        run_isolated(unittest.TestSuite, self, result)