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
class TestNonExistentConstraintMySQL(TestNonExistentConstraint, db_test_base._MySQLOpportunisticTestCase):

    def test_raise(self):
        with self.engine.connect() as conn:
            matched = self.assertRaises(exception.DBNonExistentConstraint, conn.execute, sqla.schema.DropConstraint(sqla.ForeignKeyConstraint(['id'], ['baz.id'], name='bar_fkey', table=self.table_1)))
        self.assertIsInstance(matched.inner_exception, (sqlalchemy.exc.InternalError, sqlalchemy.exc.OperationalError))
        if matched.table is not None:
            self.assertEqual('resource_foo', matched.table)
        if matched.constraint is not None:
            self.assertEqual('bar_fkey', matched.constraint)