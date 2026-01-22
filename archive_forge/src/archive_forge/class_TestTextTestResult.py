import codecs
import datetime
import doctest
import io
from itertools import chain
from itertools import combinations
import os
import platform
from queue import Queue
import re
import shutil
import sys
import tempfile
import threading
from unittest import TestSuite
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.content_type import ContentType, UTF8_TEXT
from testtools.helpers import try_import
from testtools.matchers import (
from testtools.tests.helpers import (
from testtools.testresult.doubles import (
from testtools.testresult.real import (
class TestTextTestResult(TestCase):
    """Tests for 'TextTestResult'."""

    def setUp(self):
        super().setUp()
        self.result = TextTestResult(io.StringIO())

    def getvalue(self):
        return self.result.stream.getvalue()

    def test__init_sets_stream(self):
        result = TextTestResult('fp')
        self.assertEqual('fp', result.stream)

    def reset_output(self):
        self.result.stream = io.StringIO()

    def test_startTestRun(self):
        self.result.startTestRun()
        self.assertEqual('Tests running...\n', self.getvalue())

    def test_stopTestRun_count_many(self):
        test = make_test()
        self.result.startTestRun()
        self.result.startTest(test)
        self.result.stopTest(test)
        self.result.startTest(test)
        self.result.stopTest(test)
        self.result.stream = io.StringIO()
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('\nRan 2 tests in ...s\n...', doctest.ELLIPSIS))

    def test_stopTestRun_count_single(self):
        test = make_test()
        self.result.startTestRun()
        self.result.startTest(test)
        self.result.stopTest(test)
        self.reset_output()
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('\nRan 1 test in ...s\nOK\n', doctest.ELLIPSIS))

    def test_stopTestRun_count_zero(self):
        self.result.startTestRun()
        self.reset_output()
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('\nRan 0 tests in ...s\nOK\n', doctest.ELLIPSIS))

    def test_stopTestRun_current_time(self):
        test = make_test()
        now = datetime.datetime.now(utc)
        self.result.time(now)
        self.result.startTestRun()
        self.result.startTest(test)
        now = now + datetime.timedelta(0, 0, 0, 1)
        self.result.time(now)
        self.result.stopTest(test)
        self.reset_output()
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('... in 0.001s\n...', doctest.ELLIPSIS))

    def test_stopTestRun_successful(self):
        self.result.startTestRun()
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('...\nOK\n', doctest.ELLIPSIS))

    def test_stopTestRun_not_successful_failure(self):
        test = make_failing_test()
        self.result.startTestRun()
        test.run(self.result)
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('...\nFAILED (failures=1)\n', doctest.ELLIPSIS))

    def test_stopTestRun_not_successful_error(self):
        test = make_erroring_test()
        self.result.startTestRun()
        test.run(self.result)
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('...\nFAILED (failures=1)\n', doctest.ELLIPSIS))

    def test_stopTestRun_not_successful_unexpected_success(self):
        test = make_unexpectedly_successful_test()
        self.result.startTestRun()
        test.run(self.result)
        self.result.stopTestRun()
        self.assertThat(self.getvalue(), DocTestMatches('...\nFAILED (failures=1)\n', doctest.ELLIPSIS))

    def test_stopTestRun_shows_details(self):
        self.skipTest('Disabled per bug 1188420')

        def run_tests():
            self.result.startTestRun()
            make_erroring_test().run(self.result)
            make_unexpectedly_successful_test().run(self.result)
            make_failing_test().run(self.result)
            self.reset_output()
            self.result.stopTestRun()
        run_with_stack_hidden(True, run_tests)
        self.assertThat(self.getvalue(), DocTestMatches('...======================================================================\nERROR: testtools.tests.test_testresult.Test.error\n----------------------------------------------------------------------\nTraceback (most recent call last):\n  File "...testtools...tests...test_testresult.py", line ..., in error\n    1/0\nZeroDivisionError:... divi... by zero...\n======================================================================\nFAIL: testtools.tests.test_testresult.Test.failed\n----------------------------------------------------------------------\nTraceback (most recent call last):\n  File "...testtools...tests...test_testresult.py", line ..., in failed\n    self.fail("yo!")\nAssertionError: yo!\n======================================================================\nUNEXPECTED SUCCESS: testtools.tests.test_testresult.Test.succeeded\n----------------------------------------------------------------------\n...', doctest.ELLIPSIS | doctest.REPORT_NDIFF))