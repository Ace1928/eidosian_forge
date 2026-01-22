import sys
from winappdbg import win32
from winappdbg.system import System
from winappdbg.process import Process
from winappdbg.thread import Thread
from winappdbg.module import Module
from winappdbg.window import Window
from winappdbg.breakpoint import _BreakpointContainer, CodeBreakpoint
from winappdbg.event import Event, EventHandler, EventDispatcher, EventFactory
from winappdbg.interactive import ConsoleDebugger
import warnings
def _notify_ms_vc_exception(self, event):
    """
        Notify of a Microsoft Visual C exception.

        @warning: This method is meant to be used internally by the debugger.

        @note: This allows the debugger to understand the
            Microsoft Visual C thread naming convention.

        @see: U{http://msdn.microsoft.com/en-us/library/xcb2z8hs.aspx}

        @type  event: L{ExceptionEvent}
        @param event: Microsoft Visual C exception event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    dwType = event.get_exception_information(0)
    if dwType == 4096:
        pszName = event.get_exception_information(1)
        dwThreadId = event.get_exception_information(2)
        dwFlags = event.get_exception_information(3)
        aProcess = event.get_process()
        szName = aProcess.peek_string(pszName, fUnicode=False)
        if szName:
            if dwThreadId == -1:
                dwThreadId = event.get_tid()
            if aProcess.has_thread(dwThreadId):
                aThread = aProcess.get_thread(dwThreadId)
            else:
                aThread = Thread(dwThreadId)
                aProcess._add_thread(aThread)
            aThread.set_name(szName)
    return True