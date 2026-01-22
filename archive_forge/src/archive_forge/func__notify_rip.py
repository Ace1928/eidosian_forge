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
def _notify_rip(self, event):
    """
        Notify of a RIP event.

        @warning: This method is meant to be used internally by the debugger.

        @type  event: L{RIPEvent}
        @param event: RIP event.

        @rtype:  bool
        @return: C{True} to call the user-defined handle, C{False} otherwise.
        """
    event.debug.detach(event.get_pid())
    return True