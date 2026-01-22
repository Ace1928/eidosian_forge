from winappdbg.win32.defines import *
class _SYSTEM_INFO_OEM_ID_STRUCT(Structure):
    _fields_ = [('wProcessorArchitecture', WORD), ('wReserved', WORD)]