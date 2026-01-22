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
def _encode_location_svarint(self, svarint: int) -> bytearray:
    if svarint < 0:
        return self._encode_location_varint(-svarint << 1 | 1)
    else:
        return self._encode_location_varint(svarint << 1)