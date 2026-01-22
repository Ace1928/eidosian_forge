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
def _lower_var_to_var_assign(lowerer, inst):
    """Lower Var->Var assignment.

    Returns True if-and-only-if `inst` is a Var->Var assignment.
    """
    if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Var):
        loaded = lowerer.loadvar(inst.value.name)
        lowerer.storevar(loaded, name=inst.target.name)
        return True
    return False