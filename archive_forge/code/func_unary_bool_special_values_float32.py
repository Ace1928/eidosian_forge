import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def unary_bool_special_values_float32(self, func, npfunc):
    self.unary_bool_special_values(func, npfunc, np.float32, float32)