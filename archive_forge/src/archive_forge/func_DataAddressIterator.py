import sys
import os
import ctypes
import optparse
from winappdbg import win32
from winappdbg import compat
def DataAddressIterator(memory_map):
    """
    Generator function that iterates through a memory map, returning only those
    memory blocks that contain data.

    @type  memory_map: list( L{win32.MemoryBasicInformation} )
    @param memory_map: List of memory region information objects.
        Returned by L{Process.get_memory_map}.

    @rtype:  generator of L{win32.MemoryBasicInformation}
    @return: Generator object to iterate memory blocks.
    """
    return CustomAddressIterator(memory_map, win32.MemoryBasicInformation.has_content)