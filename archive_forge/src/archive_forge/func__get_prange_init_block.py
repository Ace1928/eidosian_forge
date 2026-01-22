import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def _get_prange_init_block(self, entry_block, call_table, prange_args):
    """
        If there is init_prange, find the code between init_prange and prange
        calls. Remove the code from entry_block and return it.
        """
    init_call_ind = -1
    prange_call_ind = -1
    init_body = []
    for i, inst in enumerate(entry_block.body):
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_prange_init(inst.value.func.name, call_table):
            init_call_ind = i
        if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_parallel_loop(inst.value.func.name, call_table):
            prange_call_ind = i
    if init_call_ind != -1 and prange_call_ind != -1:
        arg_related_vars = {v.name for v in prange_args}
        saved_nodes = []
        for i in reversed(range(init_call_ind + 1, prange_call_ind)):
            inst = entry_block.body[i]
            inst_vars = {v.name for v in inst.list_vars()}
            if arg_related_vars & inst_vars:
                arg_related_vars |= inst_vars
                saved_nodes.append(inst)
            else:
                init_body.append(inst)
        init_body.reverse()
        saved_nodes.reverse()
        entry_block.body = entry_block.body[:init_call_ind] + saved_nodes + entry_block.body[prange_call_ind + 1:]
    return init_body