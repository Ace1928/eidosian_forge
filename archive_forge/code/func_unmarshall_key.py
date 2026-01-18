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
def unmarshall_key(self, key):
    """
        Unmarshalls a Crash key read from the database.

        @type  key: str or buffer
        @param key: Key to convert.

        @rtype:  L{Crash} key.
        @return: Converted key.
        """
    key = str(key)
    if self.escapeKeys:
        key = key.decode('hex')
    if self.compressKeys:
        key = zlib.decompress(key)
    key = pickle.loads(key)
    return key