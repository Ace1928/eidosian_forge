import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@cuda.jit('float32(float32, float32)', device=True)
def add2f(a, b):
    return a + b