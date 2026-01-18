import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_resultclass(self):

    def MockResultClass(*args):
        return args
    STREAM = object()
    DESCRIPTIONS = object()
    VERBOSITY = object()
    runner = unittest.TextTestRunner(STREAM, DESCRIPTIONS, VERBOSITY, resultclass=MockResultClass)
    self.assertEqual(runner.resultclass, MockResultClass)
    expectedresult = (runner.stream, DESCRIPTIONS, VERBOSITY)
    self.assertEqual(runner._makeResult(), expectedresult)