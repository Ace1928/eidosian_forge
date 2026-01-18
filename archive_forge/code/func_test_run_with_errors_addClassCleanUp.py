import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_run_with_errors_addClassCleanUp(self):
    ordering = []

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering, blowUp=True)

        def setUp(self):
            ordering.append('setUp')
            self.addCleanup(cleanup, ordering)

        def testNothing(self):
            ordering.append('test')

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'setUp', 'test', 'cleanup_good', 'tearDownClass', 'cleanup_exc'])