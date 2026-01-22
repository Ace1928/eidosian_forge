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
def _split_nonparallel_tests(test, sliced):
    """
    Split test suite into parallel and serial tests.
    """
    ptests = []
    stests = []
    flat = [*filter(sliced, _flatten_suite(test))]

    def is_parallelizable_test_case(test):
        method_name = test._testMethodName
        method = getattr(test, method_name)
        if method.__name__ != method_name and method.__name__ == 'testFailure':
            return False
        return getattr(test, '_numba_parallel_test_', True)
    for t in flat:
        if is_parallelizable_test_case(t):
            ptests.append(t)
        else:
            stests.append(t)
    return (ptests, stests)