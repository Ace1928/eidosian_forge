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
def try_fuse(equiv_set, parfor1, parfor2, metadata, func_ir, typemap):
    """try to fuse parfors and return a fused parfor, otherwise return None
    """
    dprint('try_fuse: trying to fuse \n', parfor1, '\n', parfor2)
    report = None
    if parfor1.lowerer != parfor2.lowerer:
        dprint('try_fuse: parfors different lowerers')
        msg = '- fusion failed: lowerer mismatch'
        report = FusionReport(parfor1.id, parfor2.id, msg)
        return (None, report)
    if len(parfor1.loop_nests) != len(parfor2.loop_nests):
        dprint('try_fuse: parfors number of dimensions mismatch')
        msg = '- fusion failed: number of loops mismatched, %s, %s.'
        fmt = 'parallel loop #%s has a nest of %s loops'
        l1 = fmt % (parfor1.id, len(parfor1.loop_nests))
        l2 = fmt % (parfor2.id, len(parfor2.loop_nests))
        report = FusionReport(parfor1.id, parfor2.id, msg % (l1, l2))
        return (None, report)
    ndims = len(parfor1.loop_nests)

    def is_equiv(x, y):
        return x == y or equiv_set.is_equiv(x, y)

    def get_user_varname(v):
        """get original variable name by user if possible"""
        if not isinstance(v, ir.Var):
            return v
        v = v.name
        if 'var_rename_map' in metadata and v in metadata['var_rename_map']:
            user_varname = metadata['var_rename_map'][v]
            return user_varname
        return v
    for i in range(ndims):
        nest1 = parfor1.loop_nests[i]
        nest2 = parfor2.loop_nests[i]
        if not (is_equiv(nest1.start, nest2.start) and is_equiv(nest1.stop, nest2.stop) and is_equiv(nest1.step, nest2.step)):
            dprint('try_fuse: parfor dimension correlation mismatch', i)
            msg = '- fusion failed: loop dimension mismatched in axis %s. '
            msg += 'slice(%s, %s, %s) != ' % (get_user_varname(nest1.start), get_user_varname(nest1.stop), get_user_varname(nest1.step))
            msg += 'slice(%s, %s, %s)' % (get_user_varname(nest2.start), get_user_varname(nest2.stop), get_user_varname(nest2.step))
            report = FusionReport(parfor1.id, parfor2.id, msg % i)
            return (None, report)
    func_ir._definitions = build_definitions(func_ir.blocks)
    p1_cross_dep, p1_ip, p1_ia, p1_non_ia = has_cross_iter_dep(parfor1, func_ir, typemap)
    if not p1_cross_dep:
        p2_cross_dep = has_cross_iter_dep(parfor2, func_ir, typemap, p1_ip, p1_ia, p1_non_ia)[0]
    else:
        p2_cross_dep = True
    if p1_cross_dep or p2_cross_dep:
        dprint('try_fuse: parfor cross iteration dependency found')
        msg = '- fusion failed: cross iteration dependency found between loops #%s and #%s'
        report = FusionReport(parfor1.id, parfor2.id, msg % (parfor1.id, parfor2.id))
        return (None, report)
    p1_body_usedefs = compute_use_defs(parfor1.loop_body)
    p1_body_defs = set()
    for defs in p1_body_usedefs.defmap.values():
        p1_body_defs |= defs
    p1_body_defs |= get_parfor_writes(parfor1)
    p1_body_defs |= set(parfor1.redvars)
    p2_usedefs = compute_use_defs(parfor2.loop_body)
    p2_uses = compute_use_defs({0: parfor2.init_block}).usemap[0]
    for uses in p2_usedefs.usemap.values():
        p2_uses |= uses
    overlap = p1_body_defs.intersection(p2_uses)
    if len(overlap) != 0:
        _, p2arraynotindexed = get_array_indexed_with_parfor_index(parfor2.loop_body.values(), parfor2.index_var.name, parfor2.get_loop_nest_vars(), func_ir)
        unsafe_var = (not isinstance(typemap[x], types.ArrayCompatible) or x in p2arraynotindexed for x in overlap)
        if any(unsafe_var):
            dprint('try_fuse: parfor2 depends on parfor1 body')
            msg = '- fusion failed: parallel loop %s has a dependency on the body of parallel loop %s. '
            report = FusionReport(parfor1.id, parfor2.id, msg % (parfor1.id, parfor2.id))
            return (None, report)
    return fuse_parfors_inner(parfor1, parfor2)