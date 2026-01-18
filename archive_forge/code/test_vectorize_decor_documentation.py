import numpy as np
from numba import vectorize, cuda
from numba.tests.npyufunc.test_vectorize_decor import BaseVectorizeDecor, \
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest

        Same test as .test_broadcast() but with device array as inputs
        