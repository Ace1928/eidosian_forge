from typing import Dict
from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11fnuz
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e4m3fnuz
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._custom_floats import float8_e5m2fnuz
import numpy as np
class _Float8E5m2fnuzMachArLike:

    def __init__(self):
        smallest_normal = float.fromhex('0x1p-15')
        self.smallest_normal = float8_e5m2fnuz(smallest_normal)
        smallest_subnormal = float.fromhex('0x1p-17')
        self.smallest_subnormal = float8_e5m2fnuz(smallest_subnormal)