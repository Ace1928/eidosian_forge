from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __add_running_bp(self, tid, bp):
    """Auxiliary method."""
    if tid not in self.__runningBP:
        self.__runningBP[tid] = set()
    self.__runningBP[tid].add(bp)