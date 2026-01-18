import warnings
import dis
from itertools import product
import numpy as np
from numba import njit, typed, objmode, prange
from numba.core.utils import PYVERSION
from numba.core import ir_utils, ir
from numba.core.errors import (
from numba.tests.support import (
@unittest.skipIf(PYVERSION < (3, 9), 'Python 3.9+ only')
def test_reraise_opcode_unreachable(self):

    def pyfunc():
        try:
            raise Exception
        except Exception:
            raise ValueError('ERROR')
    for inst in dis.get_instructions(pyfunc):
        if inst.opname == 'RERAISE':
            break
    else:
        self.fail('expected RERAISE opcode not found')
    func_ir = ir_utils.get_ir_of_code({}, pyfunc.__code__)
    found = False
    for lbl, blk in func_ir.blocks.items():
        for stmt in blk.find_insts(ir.StaticRaise):
            msg = 'Unreachable condition reached (op code RERAISE executed)'
            if stmt.exc_args and msg in stmt.exc_args[0]:
                found = True
    if not found:
        self.fail('expected RERAISE unreachable message not found')