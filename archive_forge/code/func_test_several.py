from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_several(self) -> None:
    """
        If several warnings are emitted by a test, C{flushWarnings} returns a
        list containing all of them.
        """
    firstMessage = 'first warning message'
    firstCategory = UserWarning
    warnings.warn(message=firstMessage, category=firstCategory)
    secondMessage = 'second warning message'
    secondCategory = RuntimeWarning
    warnings.warn(message=secondMessage, category=secondCategory)
    self.assertDictSubsets(self.flushWarnings(), [{'category': firstCategory, 'message': firstMessage}, {'category': secondCategory, 'message': secondMessage}])