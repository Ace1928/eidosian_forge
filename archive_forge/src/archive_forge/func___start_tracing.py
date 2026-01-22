from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __start_tracing(self, thread):
    """
        @type  thread: L{Thread}
        @param thread: Thread to start tracing.
        """
    tid = thread.get_tid()
    if not tid in self.__tracing:
        thread.set_tf()
        self.__tracing.add(tid)