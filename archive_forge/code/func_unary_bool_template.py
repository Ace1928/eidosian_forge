import numpy as np
from numba.cuda.testing import (skip_unless_cc_53,
from numba.np import numpy_support
from numba import cuda, float32, float64, int32, vectorize, void, int64
import math
def unary_bool_template(self, func, npfunc, npdtype, npmtype, start, stop):
    nelem = 50
    A = np.linspace(start, stop, nelem).astype(npdtype)
    B = np.empty(A.shape, dtype=np.int32)
    iarytype = npmtype[::1]
    oarytype = int32[::1]
    cfunc = cuda.jit((iarytype, oarytype))(func)
    cfunc[1, nelem](A, B)
    np.testing.assert_allclose(npfunc(A), B)