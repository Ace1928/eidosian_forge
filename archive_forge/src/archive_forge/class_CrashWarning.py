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
class CrashWarning(Warning):
    """
    An error occurred while gathering crash data.
    Some data may be incomplete or missing.
    """