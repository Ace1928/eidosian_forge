import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
class BitwiseShiftOperation(ConcreteTemplate):
    cases = [signature(max(op, types.intp), op, op2) for op in sorted(types.signed_domain) for op2 in [types.uint64, types.int64]]
    cases += [signature(max(op, types.uintp), op, op2) for op in sorted(types.unsigned_domain) for op2 in [types.uint64, types.int64]]
    unsafe_casting = False