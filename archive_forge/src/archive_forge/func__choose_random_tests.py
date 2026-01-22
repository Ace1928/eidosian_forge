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
def _choose_random_tests(tests, ratio, seed):
    """
    Choose a given proportion of tests at random.
    """
    rnd = random.Random()
    rnd.seed(seed)
    if isinstance(tests, unittest.TestSuite):
        tests = _flatten_suite(tests)
    tests = rnd.sample(tests, int(len(tests) * ratio))
    tests = sorted(tests, key=lambda case: case.id())
    return unittest.TestSuite(tests)