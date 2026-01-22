import collections
import contextlib
import cProfile
import inspect
import gc
import multiprocessing
import os
import random
import sys
import time
import unittest
import warnings
import zlib
from functools import lru_cache
from io import StringIO
from unittest import result, runner, signals, suite, loader, case
from .loader import TestLoader
from numba.core import config
class _MinimalResult(object):
    """
    A minimal, picklable TestResult-alike object.
    """
    __slots__ = ('failures', 'errors', 'skipped', 'expectedFailures', 'unexpectedSuccesses', 'stream', 'shouldStop', 'testsRun', 'test_id')

    def fixup_case(self, case):
        """
        Remove any unpicklable attributes from TestCase instance *case*.
        """
        case._outcomeForDoCleanups = None

    def __init__(self, original_result, test_id=None):
        for attr in self.__slots__:
            setattr(self, attr, getattr(original_result, attr, None))
        for case, _ in self.expectedFailures:
            self.fixup_case(case)
        for case, _ in self.errors:
            self.fixup_case(case)
        for case, _ in self.failures:
            self.fixup_case(case)
        self.test_id = test_id