import math
import random
import numpy as np
from llvmlite import ir
from numba.core.cgutils import is_nonelike
from numba.core.extending import intrinsic, overload, register_jitable
from numba.core.imputils import (Registry, impl_ret_untracked,
from numba.core.typing import signature
from numba.core import types, cgutils
from numba.np import arrayobj
from numba.core.errors import NumbaTypeError
@register_jitable
def validate_noncentral_chisquare_input(df, nonc):
    if df <= 0:
        raise ValueError('df <= 0')
    if nonc < 0:
        raise ValueError('nonc < 0')