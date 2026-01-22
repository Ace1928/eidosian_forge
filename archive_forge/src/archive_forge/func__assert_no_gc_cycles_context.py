import contextlib
import gc
import operator
import os
import platform
import pprint
import re
import shutil
import sys
import warnings
from functools import wraps
from io import StringIO
from tempfile import mkdtemp, mkstemp
from warnings import WarningMessage
import torch._numpy as np
from torch._numpy import arange, asarray as asanyarray, empty, float32, intp, ndarray
import unittest
@contextlib.contextmanager
def _assert_no_gc_cycles_context(name=None):
    __tracebackhide__ = True
    if not HAS_REFCOUNT:
        yield
        return
    assert_(gc.isenabled())
    gc.disable()
    gc_debug = gc.get_debug()
    try:
        for i in range(100):
            if gc.collect() == 0:
                break
        else:
            raise RuntimeError('Unable to fully collect garbage - perhaps a __del__ method is creating more reference cycles?')
        gc.set_debug(gc.DEBUG_SAVEALL)
        yield
        n_objects_in_cycles = gc.collect()
        objects_in_cycles = gc.garbage[:]
    finally:
        del gc.garbage[:]
        gc.set_debug(gc_debug)
        gc.enable()
    if n_objects_in_cycles:
        name_str = f' when calling {name}' if name is not None else ''
        raise AssertionError('Reference cycles were found{}: {} objects were collected, of which {} are shown below:{}'.format(name_str, n_objects_in_cycles, len(objects_in_cycles), ''.join(('\n  {} object with id={}:\n    {}'.format(type(o).__name__, id(o), pprint.pformat(o).replace('\n', '\n    ')) for o in objects_in_cycles))))