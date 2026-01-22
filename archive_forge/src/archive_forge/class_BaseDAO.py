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
class BaseDAO(object):
    """
    Data Access Object base class.

    @type _url: sqlalchemy.url.URL
    @ivar _url: Database connection URL.

    @type _dialect: str
    @ivar _dialect: SQL dialect currently being used.

    @type _driver: str
    @ivar _driver: Name of the database driver currently being used.
        To get the actual Python module use L{_url}.get_driver() instead.

    @type _session: sqlalchemy.orm.Session
    @ivar _session: Database session object.

    @type _new_session: class
    @cvar _new_session: Custom configured Session class used to create the
        L{_session} instance variable.

    @type _echo: bool
    @cvar _echo: Set to C{True} to print all SQL queries to standard output.
    """
    _echo = False
    _new_session = sessionmaker(autoflush=True, autocommit=True, expire_on_commit=True, weak_identity_map=True)

    def __init__(self, url, creator=None):
        """
        Connect to the database using the given connection URL.

        The current implementation uses SQLAlchemy and so it will support
        whatever database said module supports.

        @type  url: str
        @param url:
            URL that specifies the database to connect to.

            Some examples:
             - Opening an SQLite file:
               C{dao = CrashDAO("sqlite:///C:\\some\\path\\database.sqlite")}
             - Connecting to a locally installed SQL Express database:
               C{dao = CrashDAO("mssql://.\\SQLEXPRESS/Crashes?trusted_connection=yes")}
             - Connecting to a MySQL database running locally, using the
               C{oursql} library, authenticating as the "winappdbg" user with
               no password:
               C{dao = CrashDAO("mysql+oursql://winappdbg@localhost/Crashes")}
             - Connecting to a Postgres database running locally,
               authenticating with user and password:
               C{dao = CrashDAO("postgresql://winappdbg:winappdbg@localhost/Crashes")}

            For more information see the C{SQLAlchemy} documentation online:
            U{http://docs.sqlalchemy.org/en/latest/core/engines.html}

            Note that in all dialects except for SQLite the database
            must already exist. The tables schema, however, is created
            automatically when connecting for the first time.

            To create the database in MSSQL, you can use the
            U{SQLCMD<http://msdn.microsoft.com/en-us/library/ms180944.aspx>}
            command::
                sqlcmd -Q "CREATE DATABASE Crashes"

            In MySQL you can use something like the following::
                mysql -u root -e "CREATE DATABASE Crashes;"

            And in Postgres::
                createdb Crashes -h localhost -U winappdbg -p winappdbg -O winappdbg

            Some small changes to the schema may be tolerated (for example,
            increasing the maximum length of string columns, or adding new
            columns with default values). Of course, it's best to test it
            first before making changes in a live database. This all depends
            very much on the SQLAlchemy version you're using, but it's best
            to use the latest version always.

        @type  creator: callable
        @param creator: (Optional) Callback function that creates the SQL
            database connection.

            Normally it's not necessary to use this argument. However in some
            odd cases you may need to customize the database connection.
        """
        parsed_url = URL(url)
        schema = parsed_url.drivername
        if '+' in schema:
            dialect, driver = schema.split('+')
        else:
            dialect, driver = (schema, 'base')
        dialect = dialect.strip().lower()
        driver = driver.strip()
        arguments = {'echo': self._echo}
        if dialect == 'sqlite':
            arguments['module'] = sqlite3.dbapi2
            arguments['listeners'] = [_SQLitePatch()]
        if creator is not None:
            arguments['creator'] = creator
        engine = create_engine(url, **arguments)
        session = self._new_session(bind=engine)
        BaseDTO.metadata.create_all(engine)
        self._url = parsed_url
        self._driver = driver
        self._dialect = dialect
        self._session = session

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