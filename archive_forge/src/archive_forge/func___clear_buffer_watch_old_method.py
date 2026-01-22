from winappdbg import win32
from winappdbg import compat
import sys
from winappdbg.process import Process, Thread
from winappdbg.util import DebugRegister, MemoryAddresses
from winappdbg.textio import HexDump
import ctypes
import warnings
import traceback
def __clear_buffer_watch_old_method(self, pid, address, size):
    """
        Used by L{dont_watch_buffer} and L{dont_stalk_buffer}.

        @warn: Deprecated since WinAppDbg 1.5.

        @type  pid: int
        @param pid: Process global ID.

        @type  address: int
        @param address: Memory address of buffer to stop watching.

        @type  size: int
        @param size: Size in bytes of buffer to stop watching.
        """
    warnings.warn('Deprecated since WinAppDbg 1.5', DeprecationWarning)
    if size < 1:
        raise ValueError('Bad size for buffer watch: %r' % size)
    base = MemoryAddresses.align_address_to_page_start(address)
    limit = MemoryAddresses.align_address_to_page_end(address + size)
    pages = MemoryAddresses.get_buffer_size_in_pages(address, size)
    cset = set()
    page_addr = base
    pageSize = MemoryAddresses.pageSize
    while page_addr < limit:
        if self.has_page_breakpoint(pid, page_addr):
            bp = self.get_page_breakpoint(pid, page_addr)
            condition = bp.get_condition()
            if condition not in cset:
                if not isinstance(condition, _BufferWatchCondition):
                    continue
                cset.add(condition)
                condition.remove_last_match(address, size)
                if condition.count() == 0:
                    try:
                        self.erase_page_breakpoint(pid, bp.get_address())
                    except WindowsError:
                        pass
        page_addr = page_addr + pageSize