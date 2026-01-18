from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import numpy as np
from numba import config, cuda, njit, types
@type_callable(Interval)
def type_interval(context):

    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type
    return typer