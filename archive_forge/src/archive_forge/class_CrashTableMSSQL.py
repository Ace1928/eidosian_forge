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
class CrashTableMSSQL(CrashDictionary):
    """
    Old crash dump persistencer using a Microsoft SQL Server database.

    @warning:
        Superceded by L{CrashDictionary} since WinAppDbg 1.5.
        New applications should not use this class.
    """

    def __init__(self, location=None, allowRepeatedKeys=True):
        """
        @type  location: str
        @param location: Location of the crash database.
            It must be an ODBC connection string.

        @type  allowRepeatedKeys: bool
        @param allowRepeatedKeys:
            If C{True} all L{Crash} objects are stored.

            If C{False} any L{Crash} object with the same signature as a
            previously existing object will be ignored.
        """
        warnings.warn('The %s class is deprecated since WinAppDbg 1.5.' % self.__class__, DeprecationWarning)
        import urllib
        url = 'mssql+pyodbc:///?odbc_connect=' + urllib.quote_plus(location)
        super(CrashTableMSSQL, self).__init__(url, allowRepeatedKeys)