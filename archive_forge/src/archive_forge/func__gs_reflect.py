import numbers
from typing import List
import numpy as np
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_resize import _get_all_coords
def _gs_reflect(self, x, x_min, x_max):
    """Reflect by the near border till within the borders
        Use float for borders to avoid potential issues with integer T
        """
    fx = x
    rng = x_max - x_min
    if fx < x_min:
        dx = x_min - fx
        n = int(dx / rng)
        r = dx - n * rng
        if n % 2 == 0:
            fx = x_min + r
        else:
            fx = x_max - r
    elif fx > x_max:
        dx = fx - x_max
        n = int(dx / rng)
        r = dx - n * rng
        if n % 2 == 0:
            fx = x_max - r
        else:
            fx = x_min + r
    return fx