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
def _emit_binop_reduce_call(binop, lowerer, thread_count_var, reduce_info):
    """Emit call to the ``binop`` for the reduction variable.
    """

    def reduction_add(thread_count, redarr, init):
        c = init
        for i in range(thread_count):
            c += redarr[i]
        return c

    def reduction_mul(thread_count, redarr, init):
        c = init
        for i in range(thread_count):
            c *= redarr[i]
        return c
    kernel = {operator.iadd: reduction_add, operator.isub: reduction_add, operator.imul: reduction_mul, operator.ifloordiv: reduction_mul, operator.itruediv: reduction_mul}[binop]
    ctx = lowerer.context
    builder = lowerer.builder
    redarr_typ = reduce_info.redarr_typ
    arg_arr = lowerer.loadvar(reduce_info.redarr_var.name)
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        init_var = reduce_info.redarr_var.scope.get(reduce_info.redvar_name)
        res_print = ir.Print(args=[reduce_info.redarr_var, init_var], vararg=None, loc=lowerer.loc)
        typemap = lowerer.fndesc.typemap
        lowerer.fndesc.calltypes[res_print] = signature(types.none, typemap[reduce_info.redarr_var.name], typemap[init_var.name])
        lowerer.lower_inst(res_print)
    arg_thread_count = lowerer.loadvar(thread_count_var.name)
    args = (arg_thread_count, arg_arr, reduce_info.init_val)
    sig = signature(reduce_info.redvar_typ, types.uintp, redarr_typ, reduce_info.redvar_typ)
    redvar_result = ctx.compile_internal(builder, kernel, sig, args)
    return redvar_result