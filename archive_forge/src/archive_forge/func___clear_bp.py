from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __clear_bp(self, aThread):
    """
        Clears this breakpoint from the debug registers.

        @type  aThread: L{Thread}
        @param aThread: Thread object.
        """
    if self.__slot is not None:
        aThread.suspend()
        try:
            ctx = aThread.get_context(win32.CONTEXT_DEBUG_REGISTERS)
            DebugRegister.clear_bp(ctx, self.__slot)
            aThread.set_context(ctx)
            self.__slot = None
        finally:
            aThread.resume()