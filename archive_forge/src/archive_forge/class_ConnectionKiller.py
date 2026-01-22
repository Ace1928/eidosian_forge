from __future__ import annotations
import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref
from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal
class ConnectionKiller:

    def __init__(self):
        self.proxy_refs = weakref.WeakKeyDictionary()
        self.testing_engines = collections.defaultdict(set)
        self.dbapi_connections = set()

    def add_pool(self, pool):
        event.listen(pool, 'checkout', self._add_conn)
        event.listen(pool, 'checkin', self._remove_conn)
        event.listen(pool, 'close', self._remove_conn)
        event.listen(pool, 'close_detached', self._remove_conn)

    def _add_conn(self, dbapi_con, con_record, con_proxy):
        self.dbapi_connections.add(dbapi_con)
        self.proxy_refs[con_proxy] = True

    def _remove_conn(self, dbapi_conn, *arg):
        self.dbapi_connections.discard(dbapi_conn)

    def add_engine(self, engine, scope):
        self.add_pool(engine.pool)
        assert scope in ('class', 'global', 'function', 'fixture')
        self.testing_engines[scope].add(engine)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            warnings.warn("testing_reaper couldn't rollback/close connection: %s" % e)

    def rollback_all(self):
        for rec in list(self.proxy_refs):
            if rec is not None and rec.is_valid:
                self._safe(rec.rollback)

    def checkin_all(self):
        for rec in list(self.proxy_refs):
            if rec is not None and rec.is_valid:
                self.dbapi_connections.discard(rec.dbapi_connection)
                self._safe(rec._checkin)
        for con in self.dbapi_connections:
            self._safe(con.rollback)
        self.dbapi_connections.clear()

    def close_all(self):
        self.checkin_all()

    def prepare_for_drop_tables(self, connection):
        if not config.bootstrapped_as_sqlalchemy:
            return
        from . import provision
        provision.prepare_for_drop_tables(connection.engine.url, connection)

    def _drop_testing_engines(self, scope):
        eng = self.testing_engines[scope]
        for rec in list(eng):
            for proxy_ref in list(self.proxy_refs):
                if proxy_ref is not None and proxy_ref.is_valid:
                    if proxy_ref._pool is not None and proxy_ref._pool is rec.pool:
                        self._safe(proxy_ref._checkin)
            if hasattr(rec, 'sync_engine'):
                await_only(rec.dispose())
            else:
                rec.dispose()
        eng.clear()

    def after_test(self):
        self._drop_testing_engines('function')

    def after_test_outside_fixtures(self, test):
        if not config.bootstrapped_as_sqlalchemy:
            return
        if test.__class__.__leave_connections_for_teardown__:
            return
        self.checkin_all()
        from . import provision
        with config.db.connect() as conn:
            provision.prepare_for_drop_tables(conn.engine.url, conn)

    def stop_test_class_inside_fixtures(self):
        self.checkin_all()
        self._drop_testing_engines('function')
        self._drop_testing_engines('class')

    def stop_test_class_outside_fixtures(self):
        if pool.base._strong_ref_connection_records:
            gc_collect()
            if pool.base._strong_ref_connection_records:
                ln = len(pool.base._strong_ref_connection_records)
                pool.base._strong_ref_connection_records.clear()
                assert False, '%d connection recs not cleared after test suite' % ln

    def final_cleanup(self):
        self.checkin_all()
        for scope in self.testing_engines:
            self._drop_testing_engines(scope)

    def assert_all_closed(self):
        for rec in self.proxy_refs:
            if rec.is_valid:
                assert False