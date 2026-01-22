from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class PyDasmEngine(Engine):
    """
    Integration with PyDasm: Python bindings to libdasm.

    @see: U{https://code.google.com/p/libdasm/}
    """
    name = 'PyDasm'
    desc = 'PyDasm: Python bindings to libdasm'
    url = 'https://code.google.com/p/libdasm/'
    supported = set((win32.ARCH_I386,))

    def _import_dependencies(self):
        global pydasm
        if pydasm is None:
            import pydasm

    def decode(self, address, code):
        result = []
        offset = 0
        while offset < len(code):
            instruction = pydasm.get_instruction(code[offset:offset + 32], pydasm.MODE_32)
            current = address + offset
            if not instruction or instruction.length + offset > len(code):
                hexdump = '%.2X' % ord(code[offset])
                disasm = 'db 0x%s' % hexdump
                ilen = 1
            else:
                disasm = pydasm.get_instruction_string(instruction, pydasm.FORMAT_INTEL, current)
                ilen = instruction.length
                hexdump = HexDump.hexadecimal(code[offset:offset + ilen])
            result.append((current, ilen, disasm, hexdump))
            offset += ilen
        return result