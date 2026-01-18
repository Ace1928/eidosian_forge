import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def update_replacement_slice(self, lhs, lhs_typ, lhs_rel, dsize_rel, replacement_slice, slice_index, need_replacement, loc, scope, stmts, equiv_set, size_typ, dsize):
    known = False
    if isinstance(lhs_rel, int):
        if lhs_rel == 0:
            known = True
        elif isinstance(dsize_rel, int):
            known = True
            wil = wrap_index_literal(lhs_rel, dsize_rel)
            if wil != lhs_rel:
                if config.DEBUG_ARRAY_OPT >= 2:
                    print('Replacing slice to hard-code known slice size.')
                need_replacement = True
                literal_var, literal_typ = self.gen_literal_slice_part(wil, loc, scope, stmts, equiv_set)
                assert slice_index == 0 or slice_index == 1
                if slice_index == 0:
                    replacement_slice.args = (literal_var, replacement_slice.args[1])
                else:
                    replacement_slice.args = (replacement_slice.args[0], literal_var)
                lhs = replacement_slice.args[slice_index]
                lhs_typ = literal_typ
                lhs_rel = equiv_set.get_rel(lhs)
        elif lhs_rel < 0:
            need_replacement = True
            if config.DEBUG_ARRAY_OPT >= 2:
                print('Replacing slice due to known negative index.')
            explicit_neg_var, explicit_neg_typ = self.gen_explicit_neg(lhs, lhs_rel, lhs_typ, size_typ, loc, scope, dsize, stmts, equiv_set)
            if slice_index == 0:
                replacement_slice.args = (explicit_neg_var, replacement_slice.args[1])
            else:
                replacement_slice.args = (replacement_slice.args[0], explicit_neg_var)
            lhs = replacement_slice.args[slice_index]
            lhs_typ = explicit_neg_typ
            lhs_rel = equiv_set.get_rel(lhs)
    return (lhs, lhs_typ, lhs_rel, replacement_slice, need_replacement, known)