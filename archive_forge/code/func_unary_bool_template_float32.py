import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def unary_bool_template_float32(self, func, npfunc, start=0, stop=1):
    self.unary_template(func, npfunc, np.float32, np.float32, start, stop)