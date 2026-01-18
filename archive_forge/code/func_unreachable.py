import contextlib
import functools
from llvmlite.ir import instructions, types, values
def unreachable(self):
    inst = instructions.Unreachable(self.block)
    self._set_terminator(inst)
    return inst