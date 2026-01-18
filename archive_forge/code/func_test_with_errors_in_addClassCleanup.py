import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_with_errors_in_addClassCleanup(self):
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

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering, blowUp=True)

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    TestableTest.__module__ = 'Module'
    sys.modules['Module'] = Module
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpModule', 'setUpClass', 'test', 'tearDownClass', 'cleanup_exc', 'tearDownModule', 'cleanup_good'])