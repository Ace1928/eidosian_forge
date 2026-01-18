import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_with_errors_addCleanUp(self):
    ordering = []

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering)

        def setUp(self):
            ordering.append('setUp')
            self.addCleanup(cleanup, ordering, blowUp=True)

        def testNothing(self):
            pass

        @classmethod
        def tearDownClass(cls):
            ordering.append('tearDownClass')
    result = runTests(TestableTest)
    self.assertEqual(result.errors[0][1].splitlines()[-1], f'{CustomErrorRepr}: CleanUpExc')
    self.assertEqual(ordering, ['setUpClass', 'setUp', 'cleanup_exc', 'tearDownClass', 'cleanup_good'])