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
def test_singleFrame(self) -> None:
    """
        A C{_Traceback} object constructed with a single frame should be able
        to be passed to L{traceback.extract_tb}, and we should get a singleton
        list containing a (filename, lineno, methodname, line) tuple.
        """
    tb = failure._Traceback([], [['method', 'filename.py', 123, {}, {}]])
    self.assertEqual(traceback.extract_tb(tb), [_tb('filename.py', 123, 'method', None)])