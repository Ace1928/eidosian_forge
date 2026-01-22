from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __postCallAction_hwbp(self, event):
    """
        Handles hardware breakpoint events on return from the function.

        @type  event: L{ExceptionEvent}
        @param event: Single step event.
        """
    tid = event.get_tid()
    address = event.breakpoint.get_address()
    event.debug.erase_hardware_breakpoint(tid, address)
    try:
        self.__postCallAction(event)
    finally:
        self.__pop_params(tid)