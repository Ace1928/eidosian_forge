from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _cast_signature_pointers_to_void(self, signature):
    c_void_p = ctypes.c_void_p
    c_char_p = ctypes.c_char_p
    c_wchar_p = ctypes.c_wchar_p
    _Pointer = ctypes._Pointer
    cast = ctypes.cast
    for i in compat.xrange(len(signature)):
        t = signature[i]
        if t is not c_void_p and (issubclass(t, _Pointer) or t in [c_char_p, c_wchar_p]):
            signature[i] = cast(t, c_void_p)