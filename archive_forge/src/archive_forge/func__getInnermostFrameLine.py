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
def _getInnermostFrameLine(self, f: failure.Failure) -> str | None:
    try:
        f.raiseException()
    except ZeroDivisionError:
        tb = traceback.extract_tb(sys.exc_info()[2])
        return tb[-1].line
    else:
        raise Exception("f.raiseException() didn't raise ZeroDivisionError!?")