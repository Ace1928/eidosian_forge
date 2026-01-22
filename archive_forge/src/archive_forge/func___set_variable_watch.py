from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __set_variable_watch(self, tid, address, size, action):
    """
        Used by L{watch_variable} and L{stalk_variable}.

        @type  tid: int
        @param tid: Thread global ID.

        @type  address: int
        @param address: Memory address of variable to watch.

        @type  size: int
        @param size: Size of variable to watch. The only supported sizes are:
            byte (1), word (2), dword (4) and qword (8).

        @type  action: function
        @param action: (Optional) Action callback function.

            See L{define_hardware_breakpoint} for more details.

        @rtype:  L{HardwareBreakpoint}
        @return: Hardware breakpoint at the requested address.
        """
    if size == 1:
        sizeFlag = self.BP_WATCH_BYTE
    elif size == 2:
        sizeFlag = self.BP_WATCH_WORD
    elif size == 4:
        sizeFlag = self.BP_WATCH_DWORD
    elif size == 8:
        sizeFlag = self.BP_WATCH_QWORD
    else:
        raise ValueError('Bad size for variable watch: %r' % size)
    if self.has_hardware_breakpoint(tid, address):
        warnings.warn('Hardware breakpoint in thread %d at address %s was overwritten!' % (tid, HexDump.address(address, self.system.get_thread(tid).get_bits())), BreakpointWarning)
        bp = self.get_hardware_breakpoint(tid, address)
        if bp.get_trigger() != self.BP_BREAK_ON_ACCESS or bp.get_watch() != sizeFlag:
            self.erase_hardware_breakpoint(tid, address)
            self.define_hardware_breakpoint(tid, address, self.BP_BREAK_ON_ACCESS, sizeFlag, True, action)
            bp = self.get_hardware_breakpoint(tid, address)
    else:
        self.define_hardware_breakpoint(tid, address, self.BP_BREAK_ON_ACCESS, sizeFlag, True, action)
        bp = self.get_hardware_breakpoint(tid, address)
    return bp