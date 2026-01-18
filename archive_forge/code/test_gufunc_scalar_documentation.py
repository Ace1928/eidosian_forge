import numpy as np
from numba import guvectorize, cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
Example: sum each row using guvectorize

See Numpy documentation for detail about gufunc:
    http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
