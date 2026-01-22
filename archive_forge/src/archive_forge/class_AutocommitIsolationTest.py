import importlib
from . import testing
from .. import assert_raises
from .. import config
from .. import engines
from .. import eq_
from .. import fixtures
from .. import is_not_none
from .. import is_true
from .. import ne_
from .. import provide_metadata
from ..assertions import expect_raises
from ..assertions import expect_raises_message
from ..config import requirements
from ..provision import set_default_schema_on_connection
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import dialects
from ... import event
from ... import exc
from ... import Integer
from ... import literal_column
from ... import select
from ... import String
from ...sql.compiler import Compiled
from ...util import inspect_getfullargspec
class AutocommitIsolationTest(fixtures.TablesTest):
    run_deletes = 'each'
    __requires__ = ('autocommit',)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True, autoincrement=False), Column('data', String(50)), test_needs_acid=True)

    def _test_conn_autocommits(self, conn, autocommit):
        trans = conn.begin()
        conn.execute(self.tables.some_table.insert(), {'id': 1, 'data': 'some data'})
        trans.rollback()
        eq_(conn.scalar(select(self.tables.some_table.c.id)), 1 if autocommit else None)
        conn.rollback()
        with conn.begin():
            conn.execute(self.tables.some_table.delete())

    def test_autocommit_on(self, connection_no_trans):
        conn = connection_no_trans
        c2 = conn.execution_options(isolation_level='AUTOCOMMIT')
        self._test_conn_autocommits(c2, True)
        c2.dialect.reset_isolation_level(c2.connection.dbapi_connection)
        self._test_conn_autocommits(conn, False)

    def test_autocommit_off(self, connection_no_trans):
        conn = connection_no_trans
        self._test_conn_autocommits(conn, False)

    def test_turn_autocommit_off_via_default_iso_level(self, connection_no_trans):
        conn = connection_no_trans
        conn = conn.execution_options(isolation_level='AUTOCOMMIT')
        self._test_conn_autocommits(conn, True)
        conn.execution_options(isolation_level=requirements.get_isolation_levels(config)['default'])
        self._test_conn_autocommits(conn, False)

    @testing.requires.independent_readonly_connections
    @testing.variation('use_dialect_setting', [True, False])
    def test_dialect_autocommit_is_restored(self, testing_engine, use_dialect_setting):
        """test #10147"""
        if use_dialect_setting:
            e = testing_engine(options={'isolation_level': 'AUTOCOMMIT'})
        else:
            e = testing_engine().execution_options(isolation_level='AUTOCOMMIT')
        levels = requirements.get_isolation_levels(config)
        default = levels['default']
        with e.connect() as conn:
            self._test_conn_autocommits(conn, True)
        with e.connect() as conn:
            conn.execution_options(isolation_level=default)
            self._test_conn_autocommits(conn, False)
        with e.connect() as conn:
            self._test_conn_autocommits(conn, True)