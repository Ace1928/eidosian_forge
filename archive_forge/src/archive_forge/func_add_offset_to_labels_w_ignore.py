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
def add_offset_to_labels_w_ignore(self, blocks, offset, ignore=None):
    """add an offset to all block labels and jump/branch targets
        don't add an offset to anything in the ignore list
        """
    if ignore is None:
        ignore = set()
    new_blocks = {}
    for l, b in blocks.items():
        term = None
        if b.body:
            term = b.body[-1]
        if isinstance(term, ir.Jump):
            if term.target not in ignore:
                b.body[-1] = ir.Jump(term.target + offset, term.loc)
        if isinstance(term, ir.Branch):
            if term.truebr not in ignore:
                new_true = term.truebr + offset
            else:
                new_true = term.truebr
            if term.falsebr not in ignore:
                new_false = term.falsebr + offset
            else:
                new_false = term.falsebr
            b.body[-1] = ir.Branch(term.cond, new_true, new_false, term.loc)
        new_blocks[l + offset] = b
    return new_blocks