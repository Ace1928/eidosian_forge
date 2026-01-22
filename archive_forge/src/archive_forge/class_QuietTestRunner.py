import functools
import os
import unittest
class QuietTestRunner(object):

    def run(self, suite):
        result = unittest.TestResult()
        suite(result)
        return result