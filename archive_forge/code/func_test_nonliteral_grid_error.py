import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
@skip_on_cudasim('Cudasim does not check types')
def test_nonliteral_grid_error(self):
    with self.assertRaisesRegex(TypingError, 'RequireLiteralValue'):
        cuda.jit('void(int32)')(nonliteral_grid)