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
def test_printDetailedTracebackCapturedVarsCleaned(self) -> None:
    """
        C{printDetailedTraceback} includes information about local variables on
        the stack after C{cleanFailure} has been called.
        """
    self.assertDetailedTraceback(captureVars=True, cleanFailure=True)