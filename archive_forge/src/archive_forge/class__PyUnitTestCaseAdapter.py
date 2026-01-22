import doctest
import gc
import unittest as pyunit
from typing import Iterator, Union
from zope.interface import implementer
from twisted.python import components
from twisted.trial import itrial, reporter
from twisted.trial._synctest import _logObserver
class _PyUnitTestCaseAdapter(TestDecorator):
    """
    Adapt from pyunit.TestCase to ITestCase.
    """