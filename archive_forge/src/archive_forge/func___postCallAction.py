from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __postCallAction(self, event):
    """
        Calls the "post" callback.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint hit event.
        """
    aThread = event.get_thread()
    retval = self._get_return_value(aThread)
    self.__callHandler(self.__postCB, event, retval)