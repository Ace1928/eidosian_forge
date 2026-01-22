import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
@skip_on_cudasim('ufunc API unsupported in the simulator')
class TestVectorizeDecor(CUDATestCase, BaseVectorizeDecor):
    """
    Runs the tests from BaseVectorizeDecor with the CUDA target.
    """
    target = 'cuda'