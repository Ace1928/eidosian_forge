import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def _setup_random():
    """Sets up the deterministic random states of ``numpy`` and ``cupy``.

    """
    global _nest_count
    if _nest_count == 0:
        nondeterministic = bool(int(os.environ.get('CUPY_TEST_RANDOM_NONDETERMINISTIC', '0')))
        do_setup(not nondeterministic)
    _nest_count += 1