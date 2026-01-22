import enum
import dis
import opcode as _opcode
import sys
from marshal import dumps as _dumps
from _pydevd_frame_eval.vendored import bytecode as _bytecode
class _Variable:
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.name == other.name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '<%s %r>' % (self.__class__.__name__, self.name)