import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
class LegacyEngineFacade(object):
    """A helper class for removing of global engine instances from oslo.db.

    .. deprecated:: 1.12.0
        Please use :mod:`oslo_db.sqlalchemy.enginefacade` for new development.

    As a library, oslo.db can't decide where to store/when to create engine
    and sessionmaker instances, so this must be left for a target application.

    On the other hand, in order to simplify the adoption of oslo.db changes,
    we'll provide a helper class, which creates engine and sessionmaker
    on its instantiation and provides get_engine()/get_session() methods
    that are compatible with corresponding utility functions that currently
    exist in target projects, e.g. in Nova.

    engine/sessionmaker instances will still be global (and they are meant to
    be global), but they will be stored in the app context, rather that in the
    oslo.db context.

    Two important things to remember:

    1. An Engine instance is effectively a pool of DB connections, so it's
       meant to be shared (and it's thread-safe).
    2. A Session instance is not meant to be shared and represents a DB
       transactional context (i.e. it's not thread-safe). sessionmaker is
       a factory of sessions.

    :param sql_connection: the connection string for the database to use
    :type sql_connection: string

    :param slave_connection: the connection string for the 'slave' database
                             to use. If not provided, the master database
                             will be used for all operations. Note: this
                             is meant to be used for offloading of read
                             operations to asynchronously replicated slaves
                             to reduce the load on the master database.
    :type slave_connection: string

    :param sqlite_fk: enable foreign keys in SQLite
    :type sqlite_fk: bool

    :param expire_on_commit: expire session objects on commit
    :type expire_on_commit: bool

    Keyword arguments:

    :keyword mysql_sql_mode: the SQL mode to be used for MySQL sessions.
                             (defaults to TRADITIONAL)
    :keyword mysql_wsrep_sync_wait: value of wsrep_sync_wait for Galera
                             (defaults to None, which indicates no setting
                             will be passed)
    :keyword connection_recycle_time: Time period for connections to be
                            recycled upon checkout (defaults to 3600)
    :keyword connection_debug: verbosity of SQL debugging information.
                               -1=Off, 0=None, 100=Everything (defaults
                               to 0)
    :keyword max_pool_size: maximum number of SQL connections to keep open
                            in a pool (defaults to SQLAlchemy settings)
    :keyword max_overflow: if set, use this value for max_overflow with
                           sqlalchemy (defaults to SQLAlchemy settings)
    :keyword pool_timeout: if set, use this value for pool_timeout with
                           sqlalchemy (defaults to SQLAlchemy settings)
    :keyword sqlite_synchronous: if True, SQLite uses synchronous mode
                                 (defaults to True)
    :keyword connection_trace: add python stack traces to SQL as comment
                               strings (defaults to False)
    :keyword max_retries: maximum db connection retries during startup.
                          (setting -1 implies an infinite retry count)
                          (defaults to 10)
    :keyword retry_interval: interval between retries of opening a sql
                             connection (defaults to 10)
    :keyword thread_checkin: boolean that indicates that between each
                             engine checkin event a sleep(0) will occur to
                             allow other greenthreads to run (defaults to
                             True)

    """

    def __init__(self, sql_connection, slave_connection=None, sqlite_fk=False, expire_on_commit=False, _conf=None, _factory=None, **kwargs):
        warnings.warn('EngineFacade is deprecated; please use oslo_db.sqlalchemy.enginefacade', warning.OsloDBDeprecationWarning, stacklevel=2)
        if _factory:
            self._factory = _factory
        else:
            self._factory = _TransactionFactory()
            self._factory.configure(sqlite_fk=sqlite_fk, expire_on_commit=expire_on_commit, **kwargs)
            self._factory._start(_conf, connection=sql_connection, slave_connection=slave_connection)

    def _check_factory_started(self):
        if not self._factory._started:
            self._factory._start()

    def get_engine(self, use_slave=False):
        """Get the engine instance (note, that it's shared).

        :param use_slave: if possible, use 'slave' database for this engine.
                          If the connection string for the slave database
                          wasn't provided, 'master' engine will be returned.
                          (defaults to False)
        :type use_slave: bool

        """
        self._check_factory_started()
        if use_slave:
            return self._factory._reader_engine
        else:
            return self._factory._writer_engine

    def get_session(self, use_slave=False, **kwargs):
        """Get a Session instance.

        :param use_slave: if possible, use 'slave' database connection for
                          this session. If the connection string for the
                          slave database wasn't provided, a session bound
                          to the 'master' engine will be returned.
                          (defaults to False)
        :type use_slave: bool

        Keyword arguments will be passed to a sessionmaker instance as is (if
        passed, they will override the ones used when the sessionmaker instance
        was created). See SQLAlchemy Session docs for details.

        """
        self._check_factory_started()
        if use_slave:
            return self._factory._reader_maker(**kwargs)
        else:
            return self._factory._writer_maker(**kwargs)

    def get_sessionmaker(self, use_slave=False):
        """Get the sessionmaker instance used to create a Session.

        This can be called for those cases where the sessionmaker() is to
        be temporarily injected with some state such as a specific connection.

        """
        self._check_factory_started()
        if use_slave:
            return self._factory._reader_maker
        else:
            return self._factory._writer_maker

    @classmethod
    def from_config(cls, conf, sqlite_fk=False, expire_on_commit=False):
        """Initialize EngineFacade using oslo.config config instance options.

        :param conf: oslo.config config instance
        :type conf: oslo_config.cfg.ConfigOpts

        :param sqlite_fk: enable foreign keys in SQLite
        :type sqlite_fk: bool

        :param expire_on_commit: expire session objects on commit
        :type expire_on_commit: bool

        """
        return cls(None, sqlite_fk=sqlite_fk, expire_on_commit=expire_on_commit, _conf=conf)