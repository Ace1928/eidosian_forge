import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
def test_type_with_struct_data_model(self):

    @cuda.jit(void(test_struct_model_type[::1]))
    def f(x):
        l = cuda.local.array(10, dtype=test_struct_model_type)
        l[0] = x[0]
        x[0] = l[0]
    self.check_dtype(f, test_struct_model_type)