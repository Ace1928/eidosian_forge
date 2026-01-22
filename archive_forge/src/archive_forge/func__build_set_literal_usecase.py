import unittest
from collections import namedtuple
import contextlib
import itertools
import random
from numba.core.errors import TypingError
import numpy as np
from numba import jit, njit
from numba.tests.support import (TestCase, enable_pyobj_flags, MemoryLeakMixin,
def _build_set_literal_usecase(code, args):
    code = code % {'initializer': ', '.join((repr(arg) for arg in args))}
    return compile_function('build_set', code, globals())