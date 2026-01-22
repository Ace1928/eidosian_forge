import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from _pydevd_frame_eval.vendored import bytecode as _bytecode
from _pydevd_frame_eval.vendored.bytecode.instr import (
@staticmethod
def _pack_linetable(doff, dlineno, linetable):
    while dlineno < -127:
        linetable.append(struct.pack('Bb', 0, -127))
        dlineno -= -127
    while dlineno > 127:
        linetable.append(struct.pack('Bb', 0, 127))
        dlineno -= 127
    if doff > 254:
        linetable.append(struct.pack('Bb', 254, dlineno))
        doff -= 254
        while doff > 254:
            linetable.append(b'\xfe\x00')
            doff -= 254
        linetable.append(struct.pack('Bb', doff, 0))
    else:
        linetable.append(struct.pack('Bb', doff, dlineno))
    assert 0 <= doff <= 254
    assert -127 <= dlineno <= 127