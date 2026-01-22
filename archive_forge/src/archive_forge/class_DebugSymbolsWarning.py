from __future__ import with_statement
import sys
from winappdbg import win32
from winappdbg import compat
from winappdbg.textio import HexInput, HexDump
from winappdbg.util import PathOperations
import os
import warnings
import traceback
class DebugSymbolsWarning(UserWarning):
    """
    This warning is issued if the support for debug symbols
    isn't working properly.
    """