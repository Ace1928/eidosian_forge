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
def _assemble_exception_table(self) -> bytes:
    table = bytearray()
    for entry in self.exception_table or []:
        size = entry.stop_offset - entry.start_offset + 1
        depth = (entry.stack_depth << 1) + entry.push_lasti
        table.extend(self._encode_varint(entry.start_offset, True))
        table.extend(self._encode_varint(size))
        table.extend(self._encode_varint(entry.target))
        table.extend(self._encode_varint(depth))
    return bytes(table)