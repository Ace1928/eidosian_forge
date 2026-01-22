import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _remove_unused_temporaries(self):
    """
        Remove assignments to unused temporary variables from the
        current block.
        """
    new_body = []
    replaced_var = {}
    for inst in self.current_block.body:
        if isinstance(inst, (ir.SetItem, ir.SetAttr)) and inst.value.name in replaced_var:
            inst.value = replaced_var[inst.value.name]
        elif isinstance(inst, ir.Assign):
            if inst.target.is_temp and inst.target.name in self.assigner.unused_dests:
                continue
            if isinstance(inst.value, ir.Var) and inst.value.name in replaced_var:
                inst.value = replaced_var[inst.value.name]
                new_body.append(inst)
                continue
            if isinstance(inst.value, ir.Expr) and inst.value.op == 'exhaust_iter' and (inst.value.value.name in replaced_var):
                inst.value.value = replaced_var[inst.value.value.name]
                new_body.append(inst)
                continue
            if isinstance(inst.value, ir.Var) and inst.value.is_temp and new_body and isinstance(new_body[-1], ir.Assign):
                prev_assign = new_body[-1]
                if prev_assign.target.name == inst.value.name and (not self._var_used_in_binop(inst.target.name, prev_assign.value)):
                    replaced_var[inst.value.name] = inst.target
                    prev_assign.target = inst.target
                    self.definitions[inst.target.name].remove(inst.value)
                    self.definitions[inst.target.name].extend(self.definitions.pop(inst.value.name))
                    continue
        new_body.append(inst)
    self.current_block.body = new_body