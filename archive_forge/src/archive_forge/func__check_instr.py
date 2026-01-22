import sys
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import UNSET, Label, SetLineno, Instr
from _pydevd_frame_eval.vendored.bytecode.flags import infer_flags
def _check_instr(self, instr):
    if not isinstance(instr, (Label, SetLineno, Instr)):
        raise ValueError('Bytecode must only contain Label, SetLineno, and Instr objects, but %s was found' % type(instr).__name__)