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
def test_trapRaisesWrappedException(self) -> None:
    """
        If the wrapped C{Exception} is not a subclass of one of the
        expected types, L{failure.Failure.trap} raises the wrapped
        C{Exception}.
        """
    exception = ValueError()
    try:
        raise exception
    except BaseException:
        f = failure.Failure()
    untrapped = self.assertRaises(ValueError, f.trap, OverflowError)
    self.assertIs(exception, untrapped)