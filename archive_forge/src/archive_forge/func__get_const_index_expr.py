import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def _get_const_index_expr(stencil_ir, func_ir, index_var):
    """
    infer index_var as constant if it is of a expression form like c-1 where c
    is a constant in the outer function.
    index_var is assumed to be inside stencil kernel
    """
    const_val = guard(_get_const_index_expr_inner, stencil_ir, func_ir, index_var)
    if const_val is not None:
        return const_val
    return index_var