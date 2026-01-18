import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_with_errors_in_addCleanup(self):
    ordering = []

    class Module(object):

        @staticmethod
        def setUpModule():
            ordering.append('setUpModule')
            unittest.addModuleCleanup(cleanup, ordering)

        @staticmethod
        def tearDownModule():
            ordering.append('tearDownModule')

    class TestableTest(unittest.TestCase):

        def setUp(self):
            ordering.append('setUp')
            self.addCleanup(cleanup, ordering, blowUp=True)

        def testNothing(self):
            ordering.append('test')

        def tearDown(self):
            ordering.append('tearDown')
    TestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUp', 'test', 'tearDown', 'cleanup_exc', 'tearDownModule', 'cleanup_good'])