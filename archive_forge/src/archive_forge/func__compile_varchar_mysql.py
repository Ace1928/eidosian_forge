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
@compiles(String, 'mysql')
@compiles(VARCHAR, 'mysql')
def _compile_varchar_mysql(element, compiler, **kw):
    """MySQL hack to avoid the "VARCHAR requires a length" error."""
    if not element.length or element.length == 'max':
        return 'TEXT'
    else:
        return compiler.visit_VARCHAR(element, **kw)