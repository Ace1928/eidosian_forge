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
class _TransactionFactory:
    """A factory for :class:`._TransactionContext` objects.

    By default, there is just one of these, set up
    based on CONF, however instance-level :class:`._TransactionFactory`
    objects can be made, as is the case with the
    :class:`._TestTransactionFactory` subclass used by the oslo.db test suite.
    """

    def __init__(self):
        self._url_cfg = {'connection': _Default(), 'slave_connection': _Default()}
        self._engine_cfg = {'sqlite_fk': _Default(False), 'mysql_sql_mode': _Default('TRADITIONAL'), 'mysql_wsrep_sync_wait': _Default(), 'connection_recycle_time': _Default(3600), 'connection_debug': _Default(0), 'max_pool_size': _Default(), 'max_overflow': _Default(), 'pool_timeout': _Default(), 'sqlite_synchronous': _Default(True), 'connection_trace': _Default(False), 'max_retries': _Default(10), 'retry_interval': _Default(10), 'thread_checkin': _Default(True), 'json_serializer': _Default(None), 'json_deserializer': _Default(None), 'logging_name': _Default(None), 'connection_parameters': _Default(None)}
        self._maker_cfg = {'expire_on_commit': _Default(False)}
        self._transaction_ctx_cfg = {'rollback_reader_sessions': False, 'flush_on_subtransaction': False}
        self._facade_cfg = {'synchronous_reader': True, 'on_engine_create': []}
        self._ignored_cfg = dict(((k, _Default(None)) for k in ['db_max_retries', 'db_inc_retry_interval', 'use_db_reconnect', 'db_retry_interval', 'db_max_retry_interval', 'backend', 'use_tpool']))
        self._started = False
        self._legacy_facade = None
        self._start_lock = threading.Lock()

    def configure_defaults(self, **kw):
        """Apply default configurational options.

        This method can only be called before any specific
        transaction-beginning methods have been called.

        Configurational options are within a fixed set of keys, and fall
        under three categories: URL configuration, engine configuration,
        and session configuration.  Each key given will be tested against
        these three configuration sets to see which one is applicable; if
        it is not applicable to any set, an exception is raised.

        The configurational options given here act as **defaults**
        when the :class:`._TransactionFactory` is configured using
        a :class:`oslo_config.cfg.ConfigOpts` object; the options
        present within the :class:`oslo_config.cfg.ConfigOpts` **take
        precedence** versus the arguments passed here.  By default,
        the :class:`._TransactionFactory` loads in the configuration from
        :data:`oslo_config.cfg.CONF`, after applying the
        :data:`oslo_db.options.database_opts` configurational defaults to it.

        :param connection: database URL
        :param slave_connection: database URL
        :param sqlite_fk: whether to enable SQLite foreign key pragma; default
            False
        :param mysql_sql_mode: MySQL SQL mode, defaults to TRADITIONAL
        :param mysql_wsrep_sync_wait: MySQL wsrep_sync_wait, defaults to None,
            which indicates no setting will be passed
        :param connection_recycle_time: connection pool recycle time,
            defaults to 3600. Note the connection does not actually have to be
            "idle" to be recycled.
        :param connection_debug: engine logging level, defaults to 0. set to
            50 for INFO, 100 for DEBUG.
        :param connection_parameters: additional parameters to append onto the
            database URL query string, pass as
            "param1=value1&param2=value2&..."
        :param max_pool_size: max size of connection pool, uses CONF for
            default
        :param max_overflow: max overflow for connection pool, uses CONF for
            default
        :param sqlite_synchronous: disable SQLite SYNCHRONOUS pragma if False;
            defaults to True
        :param connection_trace: enable tracing comments in logging
        :param max_retries: max retries to connect, defaults to !0
        :param retry_interval: time in seconds between retries, defaults to 10
        :param thread_checkin: add sleep(0) on connection checkin to allow
            greenlet yields, defaults to True
        :param json_serializer: JSON serializer for PostgreSQL connections
        :param json_deserializer: JSON deserializer for PostgreSQL connections
        :param logging_name: logging name for engine
        :param expire_on_commit: sets expire_on_commit for SQLAlchemy
            sessionmaker; defaults to False
        :param rollback_reader_sessions: if True, a :class:`.Session` object
            will have its :meth:`.Session.rollback` method invoked at the end
            of a ``@reader`` block, actively rolling back the transaction and
            expiring the objects within, before the :class:`.Session` moves on
            to be closed, which has the effect of releasing connection
            resources back to the connection pool and detaching all objects.
            If False, the :class:`.Session` is not affected at the end of a
            ``@reader`` block; the underlying connection referred to by this
            :class:`.Session` will still be released in the enclosing context
            via the :meth:`.Session.close` method, which still ensures that the
            DBAPI connection is rolled back, however the objects associated
            with the :class:`.Session` retain their database-persisted contents
            after they are detached.

            .. seealso::

                http://docs.sqlalchemy.org/en/rel_0_9/glossary.html#term-released                SQLAlchemy documentation on what "releasing resources" means.
        :param synchronous_reader: whether or not to assume a "reader" context
            needs to guarantee it can read data committed by a "writer"
            assuming replication lag is present; defaults to True.  When False,
            a @reader context works the same as @async_reader and will select
            the "slave" database if present.
        :param flush_on_subtransaction: if True, a :class:`.Session` object
            will have its :meth:`.Session.flush` method invoked whenever a
            context manager or decorator that is not itself the originator of
            the top- level or savepoint :class:`.Session` transaction exits -
            in this way it behaves like a "subtransaction" from a
            :class:`.Session` perspective.

        .. seealso::

            :meth:`._TransactionFactory.configure`
        """
        self._configure(True, kw)

    def configure(self, **kw):
        """Apply configurational options.

        This method can only be called before any specific
        transaction-beginning methods have been called.

        Behavior here is the same as that of
        :meth:`._TransactionFactory.configure_defaults`,
        with the exception that values specified here will **supersede** those
        setup in the :class:`oslo_config.cfg.ConfigOpts` options.  See
        that method for a listing of all keyword arguments.

        .. seealso::

            :meth:`._TransactionFactory.configure_defaults`
        """
        self._configure(False, kw)

    def _configure(self, as_defaults, kw):
        if self._started:
            raise AlreadyStartedError('this TransactionFactory is already started')
        not_supported = []
        for k, v in kw.items():
            for dict_ in (self._url_cfg, self._engine_cfg, self._maker_cfg, self._ignored_cfg, self._facade_cfg, self._transaction_ctx_cfg):
                if k in dict_:
                    dict_[k] = _Default(v) if as_defaults else v
                    break
            else:
                not_supported.append(k)
        if not_supported:
            warnings.warn('Configuration option(s) %r not supported' % sorted(not_supported), warning.NotSupportedWarning)

    def get_legacy_facade(self):
        """Return a :class:`.LegacyEngineFacade` for this factory.

        This facade will make use of the same engine and sessionmaker
        as this factory, however will not share the same transaction context;
        the legacy facade continues to work the old way of returning
        a new Session each time get_session() is called.
        """
        if not self._legacy_facade:
            self._legacy_facade = LegacyEngineFacade(None, _factory=self)
            if not self._started:
                self._start()
        return self._legacy_facade

    def get_writer_engine(self):
        """Return the writer engine for this factory.

        Implies start.
        """
        if not self._started:
            self._start()
        return self._writer_engine

    def get_reader_engine(self):
        """Return the reader engine for this factory.

        Implies start.
        """
        if not self._started:
            self._start()
        return self._reader_engine

    def get_writer_maker(self):
        """Return the writer sessionmaker for this factory.

        Implies start.
        """
        if not self._started:
            self._start()
        return self._writer_maker

    def get_reader_maker(self):
        """Return the reader sessionmaker for this factory.

        Implies start.
        """
        if not self._started:
            self._start()
        return self._reader_maker

    def _create_connection(self, mode):
        if not self._started:
            self._start()
        if mode is _WRITER:
            return self._writer_engine.connect()
        elif mode is _ASYNC_READER or (mode is _READER and (not self.synchronous_reader)):
            return self._reader_engine.connect()
        else:
            return self._writer_engine.connect()

    def _create_session(self, mode, bind=None):
        if not self._started:
            self._start()
        kw = {}
        if bind:
            kw['bind'] = bind
        if mode is _WRITER:
            return self._writer_maker(**kw)
        elif mode is _ASYNC_READER or (mode is _READER and (not self.synchronous_reader)):
            return self._reader_maker(**kw)
        else:
            return self._writer_maker(**kw)

    def _create_factory_copy(self):
        factory = _TransactionFactory()
        factory._url_cfg.update(self._url_cfg)
        factory._engine_cfg.update(self._engine_cfg)
        factory._maker_cfg.update(self._maker_cfg)
        factory._transaction_ctx_cfg.update(self._transaction_ctx_cfg)
        factory._facade_cfg.update(self._facade_cfg)
        return factory

    def _args_for_conf(self, default_cfg, conf):
        if conf is None:
            return {key: _Default.resolve(value) for key, value in default_cfg.items() if _Default.is_set(value)}
        else:
            return {key: _Default.resolve_w_conf(value, conf, key) for key, value in default_cfg.items() if _Default.is_set_w_conf(value, conf, key)}

    def _url_args_for_conf(self, conf):
        return self._args_for_conf(self._url_cfg, conf)

    def _engine_args_for_conf(self, conf):
        return self._args_for_conf(self._engine_cfg, conf)

    def _maker_args_for_conf(self, conf):
        maker_args = self._args_for_conf(self._maker_cfg, conf)
        return maker_args

    def dispose_pool(self):
        """Call engine.pool.dispose() on underlying Engine objects."""
        with self._start_lock:
            if not self._started:
                return
            self._writer_engine.pool.dispose()
            if self._reader_engine is not self._writer_engine:
                self._reader_engine.pool.dispose()

    @property
    def is_started(self):
        """True if this :class:`._TransactionFactory` is already started."""
        return self._started

    def _start(self, conf=False, connection=None, slave_connection=None):
        with self._start_lock:
            if self._started:
                return
            if conf is False:
                conf = cfg.CONF
            if conf is not None:
                conf.register_opts(options.database_opts, 'database')
            url_args = self._url_args_for_conf(conf)
            if connection:
                url_args['connection'] = connection
            if slave_connection:
                url_args['slave_connection'] = slave_connection
            engine_args = self._engine_args_for_conf(conf)
            maker_args = self._maker_args_for_conf(conf)
            self._writer_engine, self._writer_maker = self._setup_for_connection(url_args['connection'], engine_args, maker_args)
            if url_args.get('slave_connection'):
                self._reader_engine, self._reader_maker = self._setup_for_connection(url_args['slave_connection'], engine_args, maker_args)
            else:
                self._reader_engine, self._reader_maker = (self._writer_engine, self._writer_maker)
            self.synchronous_reader = self._facade_cfg['synchronous_reader']
            self._started = True

    def _setup_for_connection(self, sql_connection, engine_kwargs, maker_kwargs):
        if sql_connection is None:
            raise exception.CantStartEngineError('No sql_connection parameter is established')
        engine = engines.create_engine(sql_connection=sql_connection, **engine_kwargs)
        for hook in self._facade_cfg['on_engine_create']:
            hook(engine)
        sessionmaker = orm.get_maker(engine=engine, **maker_kwargs)
        return (engine, sessionmaker)