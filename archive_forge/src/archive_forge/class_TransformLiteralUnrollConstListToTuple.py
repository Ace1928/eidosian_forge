from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
@register_pass(mutates_CFG=True, analysis_only=False)
class TransformLiteralUnrollConstListToTuple(FunctionPass):
    """ This pass spots a `literal_unroll([<constant values>])` and rewrites it
    as a `literal_unroll(tuple(<constant values>))`.
    """
    _name = 'transform_literal_unroll_const_list_to_tuple'
    _accepted_types = (types.BaseTuple, types.LiteralList)

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        mutated = False
        func_ir = state.func_ir
        for label, blk in func_ir.blocks.items():
            calls = [_ for _ in blk.find_exprs('call')]
            for call in calls:
                glbl = guard(get_definition, func_ir, call.func)
                if glbl and isinstance(glbl, (ir.Global, ir.FreeVar)):
                    if glbl.value is literal_unroll:
                        if len(call.args) > 1:
                            msg = 'literal_unroll takes one argument, found %s'
                            raise errors.UnsupportedError(msg % len(call.args), call.loc)
                        unroll_var = call.args[0]
                        to_unroll = guard(get_definition, func_ir, unroll_var)
                        if isinstance(to_unroll, ir.Expr) and to_unroll.op == 'build_list':
                            for i, item in enumerate(to_unroll.items):
                                val = guard(get_definition, func_ir, item)
                                if not val:
                                    msg = 'multiple definitions for variable %s, cannot resolve constant'
                                    raise errors.UnsupportedError(msg % item, to_unroll.loc)
                                if not isinstance(val, ir.Const):
                                    msg = 'Found non-constant value at position %s in a list argument to literal_unroll' % i
                                    raise errors.UnsupportedError(msg, to_unroll.loc)
                            to_unroll_lhs = guard(get_definition, func_ir, unroll_var, lhs_only=True)
                            if to_unroll_lhs is None:
                                msg = 'multiple definitions for variable %s, cannot resolve constant'
                                raise errors.UnsupportedError(msg % unroll_var, to_unroll.loc)
                            for b in func_ir.blocks.values():
                                asgn = b.find_variable_assignment(to_unroll_lhs.name)
                                if asgn is not None:
                                    break
                            else:
                                msg = 'Cannot find assignment for known variable %s' % to_unroll_lhs.name
                                raise errors.CompilerError(msg, to_unroll.loc)
                            tup = ir.Expr.build_tuple(to_unroll.items, to_unroll.loc)
                            asgn.value = tup
                            mutated = True
                        elif isinstance(to_unroll, ir.Expr) and to_unroll.op == 'build_tuple':
                            pass
                        elif isinstance(to_unroll, (ir.Global, ir.FreeVar)) and isinstance(to_unroll.value, tuple):
                            pass
                        elif isinstance(to_unroll, ir.Arg):
                            ty = state.typemap[to_unroll.name]
                            if not isinstance(ty, self._accepted_types):
                                msg = 'Invalid use of literal_unroll with a function argument, only tuples are supported as function arguments, found %s' % ty
                                raise errors.UnsupportedError(msg, to_unroll.loc)
                        else:
                            extra = None
                            if isinstance(to_unroll, ir.Expr):
                                if to_unroll.op == 'getitem':
                                    ty = state.typemap[to_unroll.value.name]
                                    if not isinstance(ty, self._accepted_types):
                                        extra = 'operation %s' % to_unroll.op
                                        loc = to_unroll.loc
                            elif isinstance(to_unroll, ir.Arg):
                                extra = 'non-const argument %s' % to_unroll.name
                                loc = to_unroll.loc
                            elif to_unroll is None:
                                extra = 'multiple definitions of variable "%s".' % unroll_var.name
                                loc = unroll_var.loc
                            else:
                                loc = to_unroll.loc
                                extra = 'unknown problem'
                            if extra:
                                msg = 'Invalid use of literal_unroll, argument should be a tuple or a list of constant values. Failure reason: found %s' % extra
                                raise errors.UnsupportedError(msg, loc)
        return mutated