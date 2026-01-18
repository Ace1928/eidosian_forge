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
def toCrash(self, getMemoryDump=False):
    """
        Returns a L{Crash} object using the data retrieved from the database.

        @type  getMemoryDump: bool
        @param getMemoryDump: If C{True} retrieve the memory dump.
            Defaults to C{False} since this may be a costly operation.

        @rtype:  L{Crash}
        @return: Crash object.
        """
    crash = Marshaller.loads(str(self.data))
    if not isinstance(crash, Crash):
        raise TypeError('Expected Crash instance, got %s instead' % type(crash))
    crash._rowid = self.id
    if not crash.memoryMap:
        memory = getattr(self, 'memory', [])
        if memory:
            crash.memoryMap = [dto.toMBI(getMemoryDump) for dto in memory]
    return crash