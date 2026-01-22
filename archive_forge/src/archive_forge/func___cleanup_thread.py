from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __cleanup_thread(self, event):
    """
        Auxiliary method for L{_notify_exit_thread}
        and L{_notify_exit_process}.
        """
    tid = event.get_tid()
    try:
        for bp in self.__runningBP[tid]:
            self.__cleanup_breakpoint(event, bp)
        del self.__runningBP[tid]
    except KeyError:
        pass
    try:
        for bp in self.__hardwareBP[tid]:
            self.__cleanup_breakpoint(event, bp)
        del self.__hardwareBP[tid]
    except KeyError:
        pass
    if tid in self.__tracing:
        self.__tracing.remove(tid)