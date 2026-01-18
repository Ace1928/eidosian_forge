import io
import sys
import textwrap
from test.support import warnings_helper, captured_stdout, captured_stderr
import traceback
import unittest
from unittest.util import strclass
def test_startTestRun_stopTestRun(self):
    result = unittest.TestResult()
    result.startTestRun()
    result.stopTestRun()