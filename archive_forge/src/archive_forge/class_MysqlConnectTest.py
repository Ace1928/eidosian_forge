import logging
import os
from unittest import mock
import fixtures
from oslo_config import cfg
import sqlalchemy
from sqlalchemy.engine import base as base_engine
from sqlalchemy import exc
from sqlalchemy.pool import NullPool
from sqlalchemy import sql
from sqlalchemy import Column, MetaData, Table
from sqlalchemy import Integer, String
from sqlalchemy.orm import declarative_base
from oslo_db import exception
from oslo_db import options as db_options
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class MysqlConnectTest(db_test_base._MySQLOpportunisticTestCase):

    def _fixture(self, sql_mode=None, mysql_wsrep_sync_wait=None):
        kw = {}
        if sql_mode is not None:
            kw['mysql_sql_mode'] = sql_mode
        if mysql_wsrep_sync_wait is not None:
            kw['mysql_wsrep_sync_wait'] = mysql_wsrep_sync_wait
        return session.create_engine(self.engine.url, **kw)

    def _assert_sql_mode(self, engine, sql_mode_present, sql_mode_non_present):
        with engine.connect() as conn:
            mode = conn.execute(sql.text("SHOW VARIABLES LIKE 'sql_mode'")).fetchone()[1]
        self.assertIn(sql_mode_present, mode)
        if sql_mode_non_present:
            self.assertNotIn(sql_mode_non_present, mode)

    def test_mysql_wsrep_sync_wait_listener(self):
        with self.engine.connect() as conn:
            try:
                conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one()
            except exc.NoResultFound:
                self.skipTest('wsrep_sync_wait option is not available')
        engine = self._fixture()
        with engine.connect() as conn:
            self.assertEqual('0', conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one())
        for wsrep_val in (2, 1, 5):
            engine = self._fixture(mysql_wsrep_sync_wait=wsrep_val)
            with engine.connect() as conn:
                self.assertEqual(str(wsrep_val), conn.execute(sql.text("show variables like '%wsrep_sync_wait%'")).scalars(1).one())

    def test_set_mode_traditional(self):
        engine = self._fixture(sql_mode='TRADITIONAL')
        self._assert_sql_mode(engine, 'TRADITIONAL', 'ANSI')

    def test_set_mode_ansi(self):
        engine = self._fixture(sql_mode='ANSI')
        self._assert_sql_mode(engine, 'ANSI', 'TRADITIONAL')

    def test_set_mode_no_mode(self):
        with self.engine.connect() as conn:
            expected = conn.execute(sql.text('SELECT @@GLOBAL.sql_mode')).scalar()
        engine = self._fixture(sql_mode=None)
        self._assert_sql_mode(engine, expected, None)

    def test_fail_detect_mode(self):
        log = self.useFixture(fixtures.FakeLogger(level=logging.WARN))
        mysql_conn = self.engine.raw_connection()
        self.addCleanup(mysql_conn.close)
        mysql_conn.detach()
        mysql_cursor = mysql_conn.cursor()

        def execute(statement, parameters=()):
            if "SHOW VARIABLES LIKE 'sql_mode'" in statement:
                statement = "SHOW VARIABLES LIKE 'i_dont_exist'"
            return mysql_cursor.execute(statement, parameters)
        test_engine = sqlalchemy.create_engine(self.engine.url, _initialize=False)
        with mock.patch.object(test_engine.pool, '_creator', mock.Mock(return_value=mock.Mock(cursor=mock.Mock(return_value=mock.Mock(execute=execute, fetchone=mysql_cursor.fetchone, fetchall=mysql_cursor.fetchall))))):
            engines._init_events.dispatch_on_drivername('mysql')(test_engine)
            test_engine.raw_connection()
        self.assertIn('Unable to detect effective SQL mode', log.output)

    def test_logs_real_mode(self):
        log = self.useFixture(fixtures.FakeLogger(level=logging.DEBUG))
        engine = self._fixture(sql_mode='TRADITIONAL')
        with engine.connect() as conn:
            actual_mode = conn.execute(sql.text("SHOW VARIABLES LIKE 'sql_mode'")).fetchone()[1]
        self.assertIn('MySQL server mode set to %s' % actual_mode, log.output)

    def test_warning_when_not_traditional(self):
        log = self.useFixture(fixtures.FakeLogger(level=logging.WARN))
        self._fixture(sql_mode='ANSI')
        self.assertIn('consider enabling TRADITIONAL or STRICT_ALL_TABLES', log.output)

    def test_no_warning_when_traditional(self):
        log = self.useFixture(fixtures.FakeLogger(level=logging.WARN))
        self._fixture(sql_mode='TRADITIONAL')
        self.assertNotIn('consider enabling TRADITIONAL or STRICT_ALL_TABLES', log.output)

    def test_no_warning_when_strict_all_tables(self):
        log = self.useFixture(fixtures.FakeLogger(level=logging.WARN))
        self._fixture(sql_mode='TRADITIONAL')
        self.assertNotIn('consider enabling TRADITIONAL or STRICT_ALL_TABLES', log.output)