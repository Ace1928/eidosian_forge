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
def _lower_non_trivial_reduce(parfor, lowerer, thread_count_var, reduce_info):
    """Lower non-trivial reduction such as call to `functools.reduce()`.
    """
    init_name = f'{reduce_info.redvar_name}#init'
    lowerer.fndesc.typemap.setdefault(init_name, reduce_info.redvar_typ)
    num_thread_llval = lowerer.loadvar(thread_count_var.name)
    with cgutils.for_range(lowerer.builder, num_thread_llval) as loop:
        tid = loop.index
        for inst in reduce_info.redvar_info.reduce_nodes:
            if _lower_var_to_var_assign(lowerer, inst):
                pass
            elif isinstance(inst, ir.Assign) and any((var.name == init_name for var in inst.list_vars())):
                elem = _emit_getitem_call(tid, lowerer, reduce_info)
                lowerer.storevar(elem, init_name)
                lowerer.lower_inst(inst)
            else:
                raise ParforsUnexpectedReduceNodeError(inst)
            if _fix_redvar_name_ssa_mismatch(parfor, lowerer, inst, reduce_info.redvar_name):
                break
    if config.DEBUG_ARRAY_OPT_RUNTIME:
        varname = reduce_info.redvar_name
        lowerer.print_variable(f'{parfor.loc}: parfor non-trivial reduction {varname} =', varname)