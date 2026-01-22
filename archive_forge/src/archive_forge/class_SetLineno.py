import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
class SetLineno:
    __slots__ = ('_lineno',)

    def __init__(self, lineno):
        _check_lineno(lineno)
        self._lineno = lineno

    @property
    def lineno(self):
        return self._lineno

    def __eq__(self, other):
        if not isinstance(other, SetLineno):
            return False
        return self._lineno == other._lineno