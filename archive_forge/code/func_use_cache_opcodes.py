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
def use_cache_opcodes(self) -> int:
    return dis._inline_cache_entries[self._opcode] if sys.version_info >= (3, 11) else 0