from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def _notify_breakpoint(self, event):
    """
        Notify breakpoints of a breakpoint exception event.

        @type  event: L{ExceptionEvent}
        @param event: Breakpoint exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    address = event.get_exception_address()
    pid = event.get_pid()
    bCallHandler = True
    key = (pid, address)
    if key in self.__codeBP:
        bp = self.__codeBP[key]
        if not bp.is_disabled():
            aThread = event.get_thread()
            aThread.set_pc(address)
            event.continueStatus = win32.DBG_CONTINUE
            bp.hit(event)
            if bp.is_running():
                tid = event.get_tid()
                self.__add_running_bp(tid, bp)
            bCondition = bp.eval_condition(event)
            if bCondition and bp.is_automatic():
                bCallHandler = bp.run_action(event)
            else:
                bCallHandler = bCondition
    elif event.get_process().is_system_defined_breakpoint(address):
        event.continueStatus = win32.DBG_CONTINUE
    elif self.in_hostile_mode():
        event.continueStatus = win32.DBG_EXCEPTION_NOT_HANDLED
    else:
        event.continueStatus = win32.DBG_CONTINUE
    return bCallHandler