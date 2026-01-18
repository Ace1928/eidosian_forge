import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_works_with_result_without_startTestRun_stopTestRun(self):

    class OldTextResult(ResultWithNoStartTestRunStopTestRun):
        separator2 = ''

        def printErrors(self):
            pass

    class Runner(unittest.TextTestRunner):

        def __init__(self):
            super(Runner, self).__init__(io.StringIO())

        def _makeResult(self):
            return OldTextResult()
    runner = Runner()
    runner.run(unittest.TestSuite())