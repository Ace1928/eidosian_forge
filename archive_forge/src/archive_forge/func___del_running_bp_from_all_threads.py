from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __del_running_bp_from_all_threads(self, bp):
    """Auxiliary method."""
    for tid, bpset in compat.iteritems(self.__runningBP):
        if bp in bpset:
            bpset.remove(bp)
            self.system.get_thread(tid).clear_tf()