import copyreg
import gc
import sys
import unittest
class LocalTestResult(unittest.TestResult):
    """A TestResult which forwards events to a parent object, except for Skips."""

    def __init__(self, parent_result):
        unittest.TestResult.__init__(self)
        self.parent_result = parent_result

    def addError(self, test, error):
        self.parent_result.addError(test, error)

    def addFailure(self, test, error):
        self.parent_result.addFailure(test, error)

    def addSkip(self, test, reason):
        pass