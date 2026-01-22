import copy
import operator
import types as pytypes
import operator
import warnings
from dataclasses import make_dataclass
import llvmlite.ir
import numpy as np
import numba
from numba.parfors import parfor
from numba.core import types, ir, config, compiler, sigutils, cgutils
from numba.core.ir_utils import (
from numba.core.typing import signature
from numba.core import lowering
from numba.parfors.parfor import ensure_parallel_support
from numba.core.errors import (
from numba.parfors.parfor_lowering_utils import ParforLoweringBuilder
def _emit_getitem_call(idx, lowerer, reduce_info):
    """Emit call to ``redarr_var[idx]``
    """

    def reducer_getitem(redarr, index):
        return redarr[index]
    builder = lowerer.builder
    ctx = lowerer.context
    redarr_typ = reduce_info.redarr_typ
    arg_arr = lowerer.loadvar(reduce_info.redarr_var.name)
    args = (arg_arr, idx)
    sig = signature(reduce_info.redvar_typ, redarr_typ, types.intp)
    elem = ctx.compile_internal(builder, reducer_getitem, sig, args)
    return elem