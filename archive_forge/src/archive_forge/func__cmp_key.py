import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
def _cmp_key(self, labels=None):
    arg = self._arg
    if self._opcode in _opcode.hasconst:
        arg = const_key(arg)
    elif isinstance(arg, Label) and labels is not None:
        arg = labels[arg]
    return (self._lineno, self._name, arg)