import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@type_callable(IndexValue)
def type_index_value(context):

    def typer(ind, mval):
        if ind == types.intp or ind == types.uintp:
            return IndexValueType(mval)
    return typer