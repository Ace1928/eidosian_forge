import cProfile as profiler
import os
import pstats
import subprocess
import sys
import numpy as np
from numba import jit
from numba.tests.support import needs_blas, expected_failure_py312
import unittest
@expected_failure_py312
def test_profiler(self):
    self.check_profiler_dot(dot)