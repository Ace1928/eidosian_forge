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
def _huntLeaks(self, test):
    self.stream.flush()
    repcount = self.repetitions
    nwarmup = self.warmup
    rc_deltas = [0] * (repcount - nwarmup)
    alloc_deltas = [0] * (repcount - nwarmup)
    _int_pool = IntPool()
    for i in range(-200, 200):
        _int_pool[i]
    for i in range(repcount):
        res = result.TestResult()
        test.run(res)
        if not res.wasSuccessful():
            self.failures.extend(res.failures)
            self.errors.extend(res.errors)
            raise AssertionError
        del res
        alloc_after, rc_after = _refleak_cleanup()
        if i >= nwarmup:
            rc_deltas[i - nwarmup] = _int_pool[rc_after - rc_before]
            alloc_deltas[i - nwarmup] = _int_pool[alloc_after - alloc_before]
        alloc_before, rc_before = (alloc_after, rc_after)
    return (rc_deltas, alloc_deltas)