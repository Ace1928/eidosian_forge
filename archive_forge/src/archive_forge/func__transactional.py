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
def _transactional(self, method, *argv, **argd):
    """
        Begins a transaction and calls the given DAO method.

        If the method executes successfully the transaction is commited.

        If the method fails, the transaction is rolled back.

        @type  method: callable
        @param method: Bound method of this class or one of its subclasses.
            The first argument will always be C{self}.

        @return: The return value of the method call.

        @raise Exception: Any exception raised by the method.
        """
    self._session.begin(subtransactions=True)
    try:
        result = method(self, *argv, **argd)
        self._session.commit()
        return result
    except:
        self._session.rollback()
        raise