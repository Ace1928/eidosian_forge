import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_startTestRun_stopTestRun_called(self):

    class LoggingTextResult(LoggingResult):
        separator2 = ''

        def printErrors(self):
            pass

    class LoggingRunner(unittest.TextTestRunner):

        def __init__(self, events):
            super(LoggingRunner, self).__init__(io.StringIO())
            self._events = events

        def _makeResult(self):
            return LoggingTextResult(self._events)
    events = []
    runner = LoggingRunner(events)
    runner.run(unittest.TestSuite())
    expected = ['startTestRun', 'stopTestRun']
    self.assertEqual(events, expected)