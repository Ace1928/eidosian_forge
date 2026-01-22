from __future__ import annotations
import sys
import traceback
import unittest as pyunit
from unittest import skipIf
from zope.interface import implementer
from twisted.python.failure import Failure
from twisted.trial.itrial import IReporter, ITestCase
from twisted.trial.test import pyunitcases
from twisted.trial.unittest import PyUnitResultAdapter, SynchronousTestCase
class PyUnitTestTests(SynchronousTestCase):

    def setUp(self) -> None:
        self.original = pyunitcases.PyUnitTest('test_pass')
        self.test = ITestCase(self.original)

    def test_callable(self) -> None:
        """
        Tests must be callable in order to be used with Python's unittest.py.
        """
        self.assertTrue(callable(self.test), f'{self.test!r} is not callable.')