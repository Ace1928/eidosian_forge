import numpy as np
from numba import float32, float64, int32, uint32
from numba.np.ufunc import Vectorize
import unittest

There was a deadlock problem when work count is smaller than number of threads.
