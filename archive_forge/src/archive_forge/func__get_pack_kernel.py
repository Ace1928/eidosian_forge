import math
import numbers
import os
import cupy
from ._util import _get_inttype
@cupy.memoize(for_each_device=True)
def _get_pack_kernel(int_type, marker=-32768):
    """Pack coordinates into array of type short2 (or int2).

    This kernel works with 2D input data, `arr` (typically boolean).

    The output array, `out` will be 3D with a signed integer dtype.
    It will have size 2 on the last axis so that it can be viewed as a CUDA
    vector type such as `int2` or `float2`.
    """
    code = f'\n    if (arr[i]) {{\n        out[2*i] = {marker};\n        out[2*i + 1] = {marker};\n    }} else {{\n        int shape_1 = arr.shape()[1];\n        int _i = i;\n        int ind_1 = _i % shape_1;\n        _i /= shape_1;\n        out[2*i] = ind_1;   // out.x\n        out[2*i + 1] = _i;  // out.y\n    }}\n    '
    return cupy.ElementwiseKernel(in_params='raw B arr', out_params='raw I out', operation=code, options=('--std=c++11',))