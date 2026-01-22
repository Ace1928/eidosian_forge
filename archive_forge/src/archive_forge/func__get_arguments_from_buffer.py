from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _get_arguments_from_buffer(self, buffer, structure):
    b_ptr = ctypes.pointer(buffer)
    v_ptr = ctypes.cast(b_ptr, ctypes.c_void_p)
    s_ptr = ctypes.cast(v_ptr, ctypes.POINTER(structure))
    struct = s_ptr.contents
    return dict([(name, struct.__getattribute__(name)) for name, type in struct._fields_])