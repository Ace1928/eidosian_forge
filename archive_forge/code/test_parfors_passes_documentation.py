import unittest
from functools import reduce
import numpy as np
from numba import njit, typeof, prange, pndindex
import numba.parfors.parfor
from numba.core import (
from numba.core.registry import cpu_target
from numba.tests.support import (TestCase, is_parfors_unsupported)

Tests for sub-components of parfors.
These tests are aimed to produce a good-enough coverage of parfor passes
so that refactoring on these passes are easier with faster testing turnaround.
