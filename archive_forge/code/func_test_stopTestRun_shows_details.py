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