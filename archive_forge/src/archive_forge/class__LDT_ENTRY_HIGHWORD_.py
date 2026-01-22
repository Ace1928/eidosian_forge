from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class _LDT_ENTRY_HIGHWORD_(Union):
    _pack_ = 1
    _fields_ = [('Bytes', _LDT_ENTRY_BYTES_), ('Bits', _LDT_ENTRY_BITS_)]