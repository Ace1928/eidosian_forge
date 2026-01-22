from typing import Dict
from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11fnuz
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e4m3fnuz
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._custom_floats import float8_e5m2fnuz
import numpy as np
@staticmethod
def _bfloat16_finfo():

    def float_to_str(f):
        return '%12.4e' % float(f)
    tiny = float.fromhex('0x1p-126')
    resolution = 0.01
    eps = float.fromhex('0x1p-7')
    epsneg = float.fromhex('0x1p-8')
    max_ = float.fromhex('0x1.FEp127')
    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max_)
    obj.min = bfloat16(-max_)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.maxexp = 128
    obj.minexp = -126
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, 'tiny'):
        obj.tiny = bfloat16(tiny)
    if not hasattr(obj, 'smallest_normal'):
        obj.smallest_normal = obj._machar.smallest_normal
    obj.smallest_subnormal = obj._machar.smallest_subnormal
    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_smallest_subnormal = float_to_str(obj.smallest_subnormal)
    obj._str_max = float_to_str(max_)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj