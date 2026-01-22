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
class MemoryDTO(BaseDTO):
    """
    Database mapping for memory dumps.
    """
    __tablename__ = 'memory'
    id = Column(Integer, Sequence(__tablename__ + '_seq'), primary_key=True, autoincrement=True)
    crash_id = Column(Integer, ForeignKey('crashes.id', ondelete='CASCADE', onupdate='CASCADE'), nullable=False)
    address = Column(BigInteger, nullable=False, index=True)
    size = Column(BigInteger, nullable=False)
    state = Column(MEM_STATE_ENUM, nullable=False)
    access = Column(MEM_ACCESS_ENUM)
    type = Column(MEM_TYPE_ENUM)
    alloc_base = Column(BigInteger)
    alloc_access = Column(MEM_ALLOC_ACCESS_ENUM)
    filename = Column(String)
    content = deferred(Column(LargeBinary))

    def __init__(self, crash_id, mbi):
        """
        Process a L{win32.MemoryBasicInformation} object for database storage.
        """
        self.crash_id = crash_id
        self.address = mbi.BaseAddress
        self.size = mbi.RegionSize
        if mbi.State == win32.MEM_RESERVE:
            self.state = 'Reserved'
        elif mbi.State == win32.MEM_COMMIT:
            self.state = 'Commited'
        elif mbi.State == win32.MEM_FREE:
            self.state = 'Free'
        else:
            self.state = 'Unknown'
        if mbi.State != win32.MEM_COMMIT:
            self.access = None
        else:
            self.access = self._to_access(mbi.Protect)
        if mbi.Type == win32.MEM_IMAGE:
            self.type = 'Image'
        elif mbi.Type == win32.MEM_MAPPED:
            self.type = 'Mapped'
        elif mbi.Type == win32.MEM_PRIVATE:
            self.type = 'Private'
        elif mbi.Type == 0:
            self.type = None
        else:
            self.type = 'Unknown'
        self.alloc_base = mbi.AllocationBase
        if not mbi.AllocationProtect:
            self.alloc_access = None
        else:
            self.alloc_access = self._to_access(mbi.AllocationProtect)
        try:
            self.filename = mbi.filename
        except AttributeError:
            self.filename = None
        try:
            self.content = mbi.content
        except AttributeError:
            self.content = None

    def _to_access(self, protect):
        if protect & win32.PAGE_NOACCESS:
            access = '--- '
        elif protect & win32.PAGE_READONLY:
            access = 'R-- '
        elif protect & win32.PAGE_READWRITE:
            access = 'RW- '
        elif protect & win32.PAGE_WRITECOPY:
            access = 'RC- '
        elif protect & win32.PAGE_EXECUTE:
            access = '--X '
        elif protect & win32.PAGE_EXECUTE_READ:
            access = 'R-X '
        elif protect & win32.PAGE_EXECUTE_READWRITE:
            access = 'RWX '
        elif protect & win32.PAGE_EXECUTE_WRITECOPY:
            access = 'RCX '
        else:
            access = '??? '
        if protect & win32.PAGE_GUARD:
            access += 'G'
        else:
            access += '-'
        if protect & win32.PAGE_NOCACHE:
            access += 'N'
        else:
            access += '-'
        if protect & win32.PAGE_WRITECOMBINE:
            access += 'W'
        else:
            access += '-'
        return access

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

    @staticmethod
    def _parse_state(state):
        if state:
            if state == 'Reserved':
                return win32.MEM_RESERVE
            if state == 'Commited':
                return win32.MEM_COMMIT
            if state == 'Free':
                return win32.MEM_FREE
        return 0

    @staticmethod
    def _parse_type(type):
        if type:
            if type == 'Image':
                return win32.MEM_IMAGE
            if type == 'Mapped':
                return win32.MEM_MAPPED
            if type == 'Private':
                return win32.MEM_PRIVATE
            return -1
        return 0

    @staticmethod
    def _parse_access(access):
        if not access:
            return 0
        perm = access[:3]
        if perm == 'R--':
            protect = win32.PAGE_READONLY
        elif perm == 'RW-':
            protect = win32.PAGE_READWRITE
        elif perm == 'RC-':
            protect = win32.PAGE_WRITECOPY
        elif perm == '--X':
            protect = win32.PAGE_EXECUTE
        elif perm == 'R-X':
            protect = win32.PAGE_EXECUTE_READ
        elif perm == 'RWX':
            protect = win32.PAGE_EXECUTE_READWRITE
        elif perm == 'RCX':
            protect = win32.PAGE_EXECUTE_WRITECOPY
        else:
            protect = win32.PAGE_NOACCESS
        if access[5] == 'G':
            protect = protect | win32.PAGE_GUARD
        if access[6] == 'N':
            protect = protect | win32.PAGE_NOCACHE
        if access[7] == 'W':
            protect = protect | win32.PAGE_WRITECOMBINE
        return protect