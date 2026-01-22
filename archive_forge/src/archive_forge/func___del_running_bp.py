from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __del_running_bp(self, tid, bp):
    """Auxiliary method."""
    self.__runningBP[tid].remove(bp)
    if not self.__runningBP[tid]:
        del self.__runningBP[tid]