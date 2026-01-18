import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def to_concrete_bytecode(self, compute_jumps_passes=None):
    converter = _bytecode._ConvertBytecodeToConcrete(self)
    return converter.to_concrete_bytecode(compute_jumps_passes=compute_jumps_passes)