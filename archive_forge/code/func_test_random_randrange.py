import collections
import functools
import math
import multiprocessing
import os
import random
import subprocess
import sys
import threading
import itertools
from textwrap import dedent
import numpy as np
import unittest
import numba
from numba import jit, _helperlib, njit
from numba.core import types
from numba.tests.support import TestCase, compile_function, tag
from numba.core.errors import TypingError
def test_random_randrange(self):
    for tp, max_width in [(types.int64, 2 ** 63), (types.int32, 2 ** 31)]:
        cf1 = njit((tp,))(random_randrange1)
        cf2 = njit((tp, tp))(random_randrange2)
        cf3 = njit((tp, tp, tp))(random_randrange3)
        self._check_randrange(cf1, cf2, cf3, get_py_state_ptr(), max_width, False)