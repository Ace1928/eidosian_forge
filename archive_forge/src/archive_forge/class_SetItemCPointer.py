import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.setitem)
class SetItemCPointer(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        ptr, idx, val = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(types.none, ptr, normalize_1d_index(idx), ptr.dtype)