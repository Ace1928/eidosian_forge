import io
import os
import sys
import pickle
import subprocess
from test import support
import unittest
from unittest.case import _Outcome
from unittest.test.support import (LoggingResult,
class TestableTest2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        ordering.append('setUpClass2')

    def testNothing(self):
        ordering.append('test2')

    @classmethod
    def tearDownClass(cls):
        ordering.append('tearDownClass2')