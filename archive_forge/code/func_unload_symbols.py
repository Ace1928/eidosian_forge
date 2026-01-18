from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
def unload_symbols(self):
    """
        Unloads the debugging symbols for all modules in this snapshot.
        """
    for aModule in self.iter_modules():
        aModule.unload_symbols()