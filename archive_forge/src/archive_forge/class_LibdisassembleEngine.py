from __future__ import with_statement
from winappdbg.textio import HexDump
from winappdbg import win32
import ctypes
import warnings
class LibdisassembleEngine(Engine):
    """
    Integration with Immunity libdisassemble.

    @see: U{http://www.immunitysec.com/resources-freesoftware.shtml}
    """
    name = 'Libdisassemble'
    desc = 'Immunity libdisassemble'
    url = 'http://www.immunitysec.com/resources-freesoftware.shtml'
    supported = set((win32.ARCH_I386,))

    def _import_dependencies(self):
        global libdisassemble
        if libdisassemble is None:
            try:
                import libdisassemble.disassemble as libdisassemble
            except ImportError:
                import disassemble as libdisassemble

    def decode(self, address, code):
        result = []
        offset = 0
        while offset < len(code):
            opcode = libdisassemble.Opcode(code[offset:offset + 32])
            length = opcode.getSize()
            disasm = opcode.printOpcode('INTEL')
            hexdump = HexDump.hexadecimal(code[offset:offset + length])
            result.append((address + offset, length, disasm, hexdump))
            offset += length
        return result