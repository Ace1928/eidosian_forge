import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def update_flags(self, *, is_async=None):
    self.flags = infer_flags(self, is_async)