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
class MockTests(SynchronousTestCase):
    """
        A test case which is used by L{FlushWarningsTests} to verify behavior
        which cannot be verified by code inside a single test method.
        """
    message = 'some warning text'
    category: type[Warning] = UserWarning

    def test_unflushed(self) -> None:
        """
            Generate a warning and don't flush it.
            """
        warnings.warn(self.message, self.category)

    def test_flushed(self) -> None:
        """
            Generate a warning and flush it.
            """
        warnings.warn(self.message, self.category)
        self.assertEqual(len(self.flushWarnings()), 1)