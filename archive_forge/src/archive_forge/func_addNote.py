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
def addNote(self, msg):
    """
        Add a note to the crash event.

        @type msg:  str
        @param msg: Note text.
        """
    self.notes.append(msg)