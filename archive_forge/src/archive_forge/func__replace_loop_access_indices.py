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
def _replace_loop_access_indices(self, loop_body, index_set, new_index):
    """
        Replace array access indices in a loop body with a new index.
        index_set has all the variables that are equivalent to loop index.
        """
    index_set.add(new_index.name)
    with dummy_return_in_loop_body(loop_body):
        labels = find_topo_order(loop_body)
    first_label = labels[0]
    added_indices = set()
    for l in labels:
        block = loop_body[l]
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                if l == first_label and stmt.value.name in index_set and (stmt.target.name not in index_set):
                    index_set.add(stmt.target.name)
                    added_indices.add(stmt.target.name)
                else:
                    scope = block.scope

                    def unver(name):
                        from numba.core import errors
                        try:
                            return scope.get_exact(name).unversioned_name
                        except errors.NotDefinedError:
                            return name
                    if unver(stmt.target.name) in map(unver, index_set) and unver(stmt.target.name) != unver(stmt.value.name):
                        raise errors.UnsupportedRewriteError('Overwrite of parallel loop index', loc=stmt.target.loc)
            if is_get_setitem(stmt):
                index = index_var_of_get_setitem(stmt)
                if index is None:
                    continue
                ind_def = guard(get_definition, self.pass_states.func_ir, index, lhs_only=True)
                if index.name in index_set or (ind_def is not None and ind_def.name in index_set):
                    set_index_var_of_get_setitem(stmt, new_index)
                guard(self._replace_multi_dim_ind, ind_def, index_set, new_index)
            if isinstance(stmt, Parfor):
                self._replace_loop_access_indices(stmt.loop_body, index_set, new_index)
    index_set -= added_indices
    return