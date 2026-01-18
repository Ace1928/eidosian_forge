import cProfile as profiler
import os
import pstats
import subprocess
import sys
import numpy as np
from numba import jit
from numba.tests.support import needs_blas, expected_failure_py312
import unittest

        Make sure the jit-compiled function shows up in the profile stats
        as a regular Python function.
        