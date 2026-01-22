from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __set_deferred_breakpoints(self, event):
    """
        Used internally. Sets all deferred breakpoints for a DLL when it's
        loaded.

        @type  event: L{LoadDLLEvent}
        @param event: Load DLL event.
        """
    pid = event.get_pid()
    try:
        deferred = self.__deferredBP[pid]
    except KeyError:
        return
    aProcess = event.get_process()
    for label, (action, oneshot) in deferred.items():
        try:
            address = aProcess.resolve_label(label)
        except Exception:
            continue
        del deferred[label]
        try:
            self.__set_break(pid, address, action, oneshot)
        except Exception:
            msg = "Can't set deferred breakpoint %s at process ID %d"
            msg = msg % (label, pid)
            warnings.warn(msg, BreakpointWarning)