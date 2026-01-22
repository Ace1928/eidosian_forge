import dis
import inspect
import opcode as _opcode
import struct
import sys
import types
from typing import (
import bytecode as _bytecode
from bytecode.flags import CompilerFlags
from bytecode.instr import (
@staticmethod
def _encode_varint(value: int, set_begin_marker: bool=False) -> Iterator[int]:
    temp: List[int] = []
    assert value >= 0
    while value:
        temp.append(value & 63 | (64 if temp else 0))
        value >>= 6
    temp = temp or [0]
    if set_begin_marker:
        temp[-1] |= 128
    return reversed(temp)