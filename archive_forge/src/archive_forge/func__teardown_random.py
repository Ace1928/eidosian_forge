import atexit
import functools
import numpy
import os
import random
import types
import unittest
import cupy
def _teardown_random():
    """Tears down the deterministic random states set up by ``_setup_random``.

    """
    global _nest_count
    assert _nest_count > 0, '_setup_random has not been called'
    _nest_count -= 1
    if _nest_count == 0:
        do_teardown()