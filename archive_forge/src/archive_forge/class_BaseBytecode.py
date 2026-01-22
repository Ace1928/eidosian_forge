import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
class BaseBytecode:

    def __init__(self):
        self.argcount = 0
        if sys.version_info > (3, 8):
            self.posonlyargcount = 0
        self.kwonlyargcount = 0
        self.first_lineno = 1
        self.name = '<module>'
        self.filename = '<string>'
        self.docstring = UNSET
        self.cellvars = []
        self.freevars = []
        self._flags = _bytecode.CompilerFlags(0)

    def _copy_attr_from(self, bytecode):
        self.argcount = bytecode.argcount
        if sys.version_info > (3, 8):
            self.posonlyargcount = bytecode.posonlyargcount
        self.kwonlyargcount = bytecode.kwonlyargcount
        self.flags = bytecode.flags
        self.first_lineno = bytecode.first_lineno
        self.name = bytecode.name
        self.filename = bytecode.filename
        self.docstring = bytecode.docstring
        self.cellvars = list(bytecode.cellvars)
        self.freevars = list(bytecode.freevars)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.argcount != other.argcount:
            return False
        if sys.version_info > (3, 8):
            if self.posonlyargcount != other.posonlyargcount:
                return False
        if self.kwonlyargcount != other.kwonlyargcount:
            return False
        if self.flags != other.flags:
            return False
        if self.first_lineno != other.first_lineno:
            return False
        if self.filename != other.filename:
            return False
        if self.name != other.name:
            return False
        if self.docstring != other.docstring:
            return False
        if self.cellvars != other.cellvars:
            return False
        if self.freevars != other.freevars:
            return False
        if self.compute_stacksize() != other.compute_stacksize():
            return False
        return True

    @property
    def flags(self):
        return self._flags

    @flags.setter
    def flags(self, value):
        if not isinstance(value, _bytecode.CompilerFlags):
            value = _bytecode.CompilerFlags(value)
        self._flags = value

    def update_flags(self, *, is_async=None):
        self.flags = infer_flags(self, is_async)