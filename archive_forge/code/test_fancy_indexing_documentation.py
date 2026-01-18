import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase

        Same as generate_advanced_index_tuples(), but also insert an
        ellipsis at various points.
        