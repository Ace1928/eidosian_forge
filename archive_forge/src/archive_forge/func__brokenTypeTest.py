from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
def _brokenTypeTest(self, detail: str) -> None:
    """
        Construct a L{Failure} with an exception type that raises an exception
        from its C{__str__} method and then call C{getTraceback} with the
        specified detail and verify that it returns a string.
        """
    f = failure.Failure(BrokenExceptionType())
    traceback = f.getTraceback(detail=detail)
    self.assertIsInstance(traceback, str)