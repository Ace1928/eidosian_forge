from winappdbg.win32.defines import *
from winappdbg.win32.version import ARCH_I386
class LDT_ENTRY(Structure):
    _pack_ = 1
    _fields_ = [('LimitLow', WORD), ('BaseLow', WORD), ('HighWord', _LDT_ENTRY_HIGHWORD_)]