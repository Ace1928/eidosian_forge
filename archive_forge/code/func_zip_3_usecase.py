import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def zip_3_usecase():
    result = 0
    for i, j, k in zip((1, 2), (3, 4, 5), (6.7, 8.9)):
        result += i * j * k
    return result