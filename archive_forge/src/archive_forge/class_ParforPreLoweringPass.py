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
class ParforPreLoweringPass(ParforPassStates):
    """ParforPreLoweringPass class is responsible for preparing parfors for lowering.
    """

    def run(self):
        """run parfor prelowering pass"""
        push_call_vars(self.func_ir.blocks, {}, {}, self.typemap)
        dprint_func_ir(self.func_ir, 'after push call vars')
        simplify(self.func_ir, self.typemap, self.calltypes, self.metadata['parfors'])
        dprint_func_ir(self.func_ir, 'after optimization')
        if config.DEBUG_ARRAY_OPT >= 1:
            print('variable types: ', sorted(self.typemap.items()))
            print('call types: ', self.calltypes)
        if config.DEBUG_ARRAY_OPT >= 3:
            for block_label, block in self.func_ir.blocks.items():
                new_block = []
                scope = block.scope
                for stmt in block.body:
                    new_block.append(stmt)
                    if isinstance(stmt, ir.Assign):
                        loc = stmt.loc
                        lhs = stmt.target
                        rhs = stmt.value
                        lhs_typ = self.typemap[lhs.name]
                        print('Adding print for assignment to ', lhs.name, lhs_typ, type(lhs_typ))
                        if lhs_typ in types.number_domain or isinstance(lhs_typ, types.Literal):
                            str_var = ir.Var(scope, mk_unique_var('str_var'), loc)
                            self.typemap[str_var.name] = types.StringLiteral(lhs.name)
                            lhs_const = ir.Const(lhs.name, loc)
                            str_assign = ir.Assign(lhs_const, str_var, loc)
                            new_block.append(str_assign)
                            str_print = ir.Print([str_var], None, loc)
                            self.calltypes[str_print] = signature(types.none, self.typemap[str_var.name])
                            new_block.append(str_print)
                            ir_print = ir.Print([lhs], None, loc)
                            self.calltypes[ir_print] = signature(types.none, lhs_typ)
                            new_block.append(ir_print)
                block.body = new_block
        if self.func_ir.is_generator:
            fix_generator_types(self.func_ir.generator_info, self.return_type, self.typemap)
        if sequential_parfor_lowering:
            lower_parfor_sequential(self.typingctx, self.func_ir, self.typemap, self.calltypes, self.metadata)
        else:
            parfor_ids, parfors = get_parfor_params(self.func_ir.blocks, self.options.fusion, self.nested_fusion_info)
            for p in parfors:
                p.redvars, p.reddict = get_parfor_reductions(self.func_ir, p, p.params, self.calltypes)
            for p in parfors:
                p.validate_params(self.typemap)
            if config.DEBUG_ARRAY_OPT_STATS:
                name = self.func_ir.func_id.func_qualname
                n_parfors = len(parfor_ids)
                if n_parfors > 0:
                    after_fusion = 'After fusion' if self.options.fusion else 'With fusion disabled'
                    print('{}, function {} has {} parallel for-loop(s) #{}.'.format(after_fusion, name, n_parfors, parfor_ids))
                else:
                    print('Function {} has no Parfor.'.format(name))