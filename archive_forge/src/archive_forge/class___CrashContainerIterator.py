from winappdbg import win32
from winappdbg import compat
from winappdbg.system import System
from winappdbg.textio import HexDump, CrashDump
from winappdbg.util import StaticClass, MemoryAddresses, PathOperations
import sys
import os
import time
import zlib
import warnings
class __CrashContainerIterator(object):
    """
        Iterator of Crash objects. Returned by L{CrashContainer.__iter__}.
        """

    def __init__(self, container):
        """
            @type  container: L{CrashContainer}
            @param container: Crash set to iterate.
            """
        self.__container = container
        self.__keys_iter = compat.iterkeys(container)

    def next(self):
        """
            @rtype:  L{Crash}
            @return: A B{copy} of a Crash object in the L{CrashContainer}.
            @raise StopIteration: No more items left.
            """
        key = self.__keys_iter.next()
        return self.__container.get(key)