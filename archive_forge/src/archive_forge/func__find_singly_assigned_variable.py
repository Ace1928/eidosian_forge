from collections import namedtuple, defaultdict
import operator
import warnings
from functools import partial
import llvmlite.ir
from llvmlite.ir import Constant, IRBuilder
from numba.core import (typing, utils, types, ir, debuginfo, funcdesc,
from numba.core.errors import (LoweringError, new_error_context, TypingError,
from numba.core.funcdesc import default_mangler
from numba.core.environment import Environment
from numba.core.analysis import compute_use_defs, must_use_alloca
from numba.misc.firstlinefinder import get_func_body_first_lineno
def _find_singly_assigned_variable(self):
    func_ir = self.func_ir
    blocks = func_ir.blocks
    sav = set()
    if not self.func_ir.func_id.is_generator:
        use_defs = compute_use_defs(blocks)
        alloca_vars = must_use_alloca(blocks)
        var_assign_map = defaultdict(set)
        for blk, vl in use_defs.defmap.items():
            for var in vl:
                var_assign_map[var].add(blk)
        var_use_map = defaultdict(set)
        for blk, vl in use_defs.usemap.items():
            for var in vl:
                var_use_map[var].add(blk)
        for var in var_assign_map:
            if var not in alloca_vars and len(var_assign_map[var]) == 1:
                if len(var_use_map[var]) == 0:
                    [defblk] = var_assign_map[var]
                    assign_stmts = self.blocks[defblk].find_insts(ir.Assign)
                    assigns = [stmt for stmt in assign_stmts if stmt.target.name == var]
                    if len(assigns) == 1:
                        sav.add(var)
    self._singly_assigned_vars = sav
    self._blk_local_varmap = {}