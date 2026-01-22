import contextlib
import itertools
from unittest import mock
import sqlalchemy as sqla
from sqlalchemy import event
import sqlalchemy.exc
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy import sql
from oslo_db import exception
from oslo_db.sqlalchemy import compat
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import exc_filters
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
from oslo_db.tests import utils as test_utils
class TestExceptionCauseMySQLSavepoint(db_test_base._MySQLOpportunisticTestCase):

    def setUp(self):
        super(TestExceptionCauseMySQLSavepoint, self).setUp()
        Base = declarative_base()

        class A(Base):
            __tablename__ = 'a'
            id = sqla.Column(sqla.Integer, primary_key=True)
            __table_args__ = {'mysql_engine': 'InnoDB'}
        Base.metadata.create_all(self.engine)
        self.A = A

    def test_cause_for_failed_flush_plus_no_savepoint(self):
        session = self.sessionmaker()
        with session.begin():
            session.add(self.A(id=1))
        try:
            with session.begin():
                try:
                    with session.begin_nested():
                        session.execute(sql.text('rollback'))
                        session.add(self.A(id=1))
                except exception.DBError as dbe_inner:
                    self.assertIsInstance(dbe_inner.cause, exception.DBDuplicateEntry)
        except exception.DBError as dbe_outer:
            self.AssertIsInstance(dbe_outer.cause, exception.DBDuplicateEntry)
        try:
            with session.begin():
                session.add(self.A(id=1))
        except exception.DBError as dbe_outer:
            self.assertIsNone(dbe_outer.cause)

    def test_rollback_doesnt_interfere_with_killed_conn(self):
        session = self.sessionmaker()
        session.begin()
        try:
            session.execute(sql.text('select 1'))
            compat.driver_connection(session.connection()).close()
            session.execute(sql.text('select 1'))
        except exception.DBConnectionError:
            session.rollback()
        else:
            assert False, 'no exception raised'

    def test_savepoint_rollback_doesnt_interfere_with_killed_conn(self):
        session = self.sessionmaker()
        session.begin()
        try:
            session.begin_nested()
            session.execute(sql.text('select 1'))
            compat.driver_connection(session.connection()).close()
            session.execute(sql.text('select 1'))
        except exception.DBConnectionError:
            session.rollback()
        else:
            assert False, 'no exception raised'