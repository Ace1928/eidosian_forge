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
class CrashTable(CrashDictionary):
    """
    Old crash dump persistencer using a SQLite database.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, location=None, allowRepeatedKeys=True):
        """
        @type  location: str
        @param location: (Optional) Location of the crash database.
            If the location is a filename, it's an SQLite database file.

            If no location is specified, the container is volatile.
            Volatile containers are stored only in memory and
            destroyed when they go out of scope.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        warnings.warn('The %s class is deprecated since WinAppDbg 1.5.' % self.__class__, DeprecationWarning)
        if location:
            url = 'sqlite:///%s' % location
        else:
            url = 'sqlite://'
        super(CrashTable, self).__init__(url, allowRepeatedKeys)