import copy
import functools
import threading
import time
from oslo_utils import strutils
import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import pool as sa_pool
from sqlalchemy import sql
import tenacity
from taskflow import exceptions as exc
from taskflow import logging
from taskflow.persistence.backends.sqlalchemy import migration
from taskflow.persistence.backends.sqlalchemy import tables
from taskflow.persistence import base
from taskflow.persistence import models
from taskflow.utils import eventlet_utils
from taskflow.utils import misc
class SQLAlchemyBackend(base.Backend):
    """A sqlalchemy backend.

    Example configuration::

        conf = {
            "connection": "sqlite:////tmp/test.db",
        }
    """

    def __init__(self, conf, engine=None):
        super(SQLAlchemyBackend, self).__init__(conf)
        if engine is not None:
            self._engine = engine
            self._owns_engine = False
        else:
            self._engine = self._create_engine(self._conf)
            self._owns_engine = True
        self._validated = False
        self._upgrade_lock = threading.Lock()
        try:
            self._max_retries = misc.as_int(self._conf.get('max_retries'))
        except TypeError:
            self._max_retries = 0

    @staticmethod
    def _create_engine(conf):
        conf = copy.deepcopy(conf)
        engine_args = {'echo': _as_bool(conf.pop('echo', False)), 'pool_recycle': 3600}
        if 'idle_timeout' in conf:
            idle_timeout = misc.as_int(conf.pop('idle_timeout'))
            engine_args['pool_recycle'] = idle_timeout
        sql_connection = conf.pop('connection')
        e_url = sa.engine.url.make_url(sql_connection)
        if 'sqlite' in e_url.drivername:
            engine_args['poolclass'] = sa_pool.NullPool
            if sql_connection.lower().strip() in SQLITE_IN_MEMORY:
                engine_args['poolclass'] = sa_pool.StaticPool
                engine_args['connect_args'] = {'check_same_thread': False}
        else:
            for k, lookup_key in [('pool_size', 'max_pool_size'), ('max_overflow', 'max_overflow'), ('pool_timeout', 'pool_timeout')]:
                if lookup_key in conf:
                    engine_args[k] = misc.as_int(conf.pop(lookup_key))
        if 'isolation_level' not in conf:
            txn_isolation_levels = conf.pop('isolation_levels', DEFAULT_TXN_ISOLATION_LEVELS)
            level_applied = False
            for driver, level in txn_isolation_levels.items():
                if driver == e_url.drivername:
                    engine_args['isolation_level'] = level
                    level_applied = True
                    break
            if not level_applied:
                for driver, level in txn_isolation_levels.items():
                    if e_url.drivername.find(driver) != -1:
                        engine_args['isolation_level'] = level
                        break
        else:
            engine_args['isolation_level'] = conf.pop('isolation_level')
        engine_args.update(conf.pop('engine_args', {}))
        engine = sa.create_engine(sql_connection, **engine_args)
        log_statements = conf.pop('log_statements', False)
        if _as_bool(log_statements):
            log_statements_level = conf.pop('log_statements_level', logging.TRACE)
            sa.event.listen(engine, 'before_cursor_execute', functools.partial(_log_statements, log_statements_level))
        checkin_yield = conf.pop('checkin_yield', eventlet_utils.EVENTLET_AVAILABLE)
        if _as_bool(checkin_yield):
            sa.event.listen(engine, 'checkin', _thread_yield)
        if 'mysql' in e_url.drivername:
            if _as_bool(conf.pop('checkout_ping', True)):
                sa.event.listen(engine, 'checkout', _ping_listener)
            mode = None
            if 'mysql_sql_mode' in conf:
                mode = conf.pop('mysql_sql_mode')
            if mode is not None:
                sa.event.listen(engine, 'connect', functools.partial(_set_sql_mode, mode))
        return engine

    @property
    def engine(self):
        return self._engine

    def get_connection(self):
        conn = Connection(self, upgrade_lock=self._upgrade_lock)
        if not self._validated:
            conn.validate(max_retries=self._max_retries)
            self._validated = True
        return conn

    def close(self):
        if self._owns_engine:
            self._engine.dispose()
        self._validated = False