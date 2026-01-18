import sqlite3
import datetime
import warnings
from sqlalchemy import create_engine, Column, ForeignKey, Sequence
from sqlalchemy.engine.url import URL
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.interfaces import PoolListener
from sqlalchemy.orm import sessionmaker, deferred
from sqlalchemy.orm.exc import NoResultFound, MultipleResultsFound
from sqlalchemy.types import Integer, BigInteger, Boolean, DateTime, String, \
from sqlalchemy.sql.expression import asc, desc
from crash import Crash, Marshaller, pickle, HIGHEST_PROTOCOL
from textio import CrashDump
import win32
def toMBI(self, getMemoryDump=False):
    """
        Returns a L{win32.MemoryBasicInformation} object using the data
        retrieved from the database.

        @type  getMemoryDump: bool
        @param getMemoryDump: (Optional) If C{True} retrieve the memory dump.
            Defaults to C{False} since this may be a costly operation.

        @rtype:  L{win32.MemoryBasicInformation}
        @return: Memory block information.
        """
    mbi = win32.MemoryBasicInformation()
    mbi.BaseAddress = self.address
    mbi.RegionSize = self.size
    mbi.State = self._parse_state(self.state)
    mbi.Protect = self._parse_access(self.access)
    mbi.Type = self._parse_type(self.type)
    if self.alloc_base is not None:
        mbi.AllocationBase = self.alloc_base
    else:
        mbi.AllocationBase = mbi.BaseAddress
    if self.alloc_access is not None:
        mbi.AllocationProtect = self._parse_access(self.alloc_access)
    else:
        mbi.AllocationProtect = mbi.Protect
    if self.filename is not None:
        mbi.filename = self.filename
    if getMemoryDump and self.content is not None:
        mbi.content = self.content
    return mbi