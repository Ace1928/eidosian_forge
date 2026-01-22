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
class TestLister(object):
    """Simply list available tests rather than running them."""

    def __init__(self, useslice):
        self.useslice = parse_slice(useslice)

    def run(self, test):
        result = runner.TextTestResult(sys.stderr, descriptions=True, verbosity=1)
        self._test_list = _flatten_suite(test)
        masked_list = list(filter(self.useslice, self._test_list))
        self._test_list.sort(key=cuda_sensitive_mtime)
        for t in masked_list:
            print(t.id())
        print('%d tests found. %s selected' % (len(self._test_list), len(masked_list)))
        return result