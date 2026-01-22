from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __cleanup_breakpoint(self, event, bp):
    """Auxiliary method."""
    try:
        process = event.get_process()
        thread = event.get_thread()
        bp.disable(process, thread)
    except Exception:
        pass
    bp.set_condition(True)
    bp.set_action(None)