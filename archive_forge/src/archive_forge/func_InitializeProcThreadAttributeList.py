import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def InitializeProcThreadAttributeList(dwAttributeCount):
    _InitializeProcThreadAttributeList = windll.kernel32.InitializeProcThreadAttributeList
    _InitializeProcThreadAttributeList.argtypes = [LPPROC_THREAD_ATTRIBUTE_LIST, DWORD, DWORD, PSIZE_T]
    _InitializeProcThreadAttributeList.restype = bool
    Size = SIZE_T(0)
    _InitializeProcThreadAttributeList(None, dwAttributeCount, 0, byref(Size))
    RaiseIfZero(Size.value)
    AttributeList = (BYTE * Size.value)()
    success = _InitializeProcThreadAttributeList(byref(AttributeList), dwAttributeCount, 0, byref(Size))
    RaiseIfZero(success)
    return AttributeList