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
def _parfor_lowering_finalize_reduction(parfor, redarrs, lowerer, parfor_reddict, thread_count_var):
    """Emit code to finalize the reduction from the intermediate values of
    each thread.
    """
    for redvar_name, redarr_var in redarrs.items():
        redvar_typ = lowerer.fndesc.typemap[redvar_name]
        redarr_typ = lowerer.fndesc.typemap[redarr_var.name]
        init_val = lowerer.loadvar(redvar_name)
        reduce_info = _ReductionInfo(redvar_info=parfor_reddict[redvar_name], redvar_name=redvar_name, redvar_typ=redvar_typ, redarr_var=redarr_var, redarr_typ=redarr_typ, init_val=init_val)
        handler = _lower_trivial_inplace_binops if reduce_info.redvar_info.redop is not None else _lower_non_trivial_reduce
        handler(parfor, lowerer, thread_count_var, reduce_info)