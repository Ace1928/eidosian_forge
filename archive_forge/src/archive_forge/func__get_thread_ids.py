from __future__ import with_statement
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexDump
from winappdbg.util import DebugRegister
from winappdbg.window import Window
import sys
import struct
import warnings
def _get_thread_ids(self):
    """
        Private method to get the list of thread IDs currently in the snapshot
        without triggering an automatic scan.
        """
    return compat.keys(self.__threadDict)