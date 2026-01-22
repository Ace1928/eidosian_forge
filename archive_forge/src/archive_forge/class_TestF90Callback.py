import math
import textwrap
import sys
import pytest
import threading
import traceback
import time
import numpy as np
from numpy.testing import IS_PYPY
from . import util
class TestF90Callback(util.F2PyTest):
    sources = [util.getpath('tests', 'src', 'callback', 'gh17797.f90')]

    def test_gh17797(self):

        def incr(x):
            return x + 123
        y = np.array([1, 2, 3], dtype=np.int64)
        r = self.module.gh17797(incr, y)
        assert r == 123 + 1 + 2 + 3