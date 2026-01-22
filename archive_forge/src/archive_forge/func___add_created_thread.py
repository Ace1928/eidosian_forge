from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def __add_created_thread(self, event):
    """
        Private method to automatically add new thread objects from debug events.

        @type  event: L{Event}
        @param event: Event object.
        """
    dwThreadId = event.get_tid()
    hThread = event.get_thread_handle()
    if not self._has_thread_id(dwThreadId):
        aThread = Thread(dwThreadId, hThread, self)
        teb_ptr = event.get_teb()
        if teb_ptr:
            aThread._teb_ptr = teb_ptr
        self._add_thread(aThread)