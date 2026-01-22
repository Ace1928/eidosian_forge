import doctest
import gc
import unittest as pyunit
from typing import Iterator, Union
from zope.interface import implementer
from twisted.python import components
from twisted.trial import itrial, reporter
from twisted.trial._synctest import _logObserver
class _ForceGarbageCollectionDecorator(TestDecorator):
    """
    Forces garbage collection to be run before and after the test. Any errors
    logged during the post-test collection are added to the test result as
    errors.
    """

    def run(self, result):
        gc.collect()
        TestDecorator.run(self, result)
        _logObserver._add()
        gc.collect()
        for error in _logObserver.getErrors():
            result.addError(self, error)
        _logObserver.flushErrors()
        _logObserver._remove()