import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@infer_global(operator.getitem)
class GetItemCPointer(AbstractTemplate):

    def generic(self, args, kws):
        assert not kws
        ptr, idx = args
        if isinstance(ptr, types.CPointer) and isinstance(idx, types.Integer):
            return signature(ptr.dtype, ptr, normalize_1d_index(idx))