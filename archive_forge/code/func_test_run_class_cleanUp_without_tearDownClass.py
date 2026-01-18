import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
def test_run_class_cleanUp_without_tearDownClass(self):
    ordering = []
    blowUp = True

    class TestableTest(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            ordering.append('setUpClass')
            cls.addClassCleanup(cleanup, ordering)
            if blowUp:
                raise CustomError()

        def testNothing(self):
            ordering.append('test')

        @classmethod
        @property
        def tearDownClass(cls):
            raise AttributeError
    runTests(TestableTest)
    self.assertEqual(ordering, ['setUpClass', 'cleanup_good'])
    ordering = []
    blowUp = False
    runTests(TestableTest)
    self.assertEqual(ordering, ['setUpClass', 'test', 'cleanup_good'])