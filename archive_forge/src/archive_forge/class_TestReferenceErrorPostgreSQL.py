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
class TestReferenceErrorPostgreSQL(TestReferenceErrorSQLite, db_test_base._PostgreSQLOpportunisticTestCase):

    def test_raise(self):
        with self.engine.connect() as conn:
            params = {'id': 1, 'foo_id': 2}
            matched = self.assertRaises(exception.DBReferenceError, conn.execute, self.table_2.insert().values(**params))
        self.assertInnerException(matched, 'IntegrityError', 'insert or update on table "resource_entity" violates foreign key constraint "foo_fkey"\nDETAIL:  Key (foo_id)=(2) is not present in table "resource_foo".\n', 'INSERT INTO resource_entity (id, foo_id) VALUES (%(id)s, %(foo_id)s)', params)
        self.assertEqual('resource_entity', matched.table)
        self.assertEqual('foo_fkey', matched.constraint)
        self.assertEqual('foo_id', matched.key)
        self.assertEqual('resource_foo', matched.key_table)

    def test_raise_delete(self):
        with self.engine.connect() as conn:
            with conn.begin():
                conn.execute(self.table_1.insert().values(id=1234, foo=42))
                conn.execute(self.table_2.insert().values(id=4321, foo_id=1234))
            with conn.begin():
                matched = self.assertRaises(exception.DBReferenceError, conn.execute, self.table_1.delete())
        self.assertInnerException(matched, 'IntegrityError', 'update or delete on table "resource_foo" violates foreign key constraint "foo_fkey" on table "resource_entity"\nDETAIL:  Key (id)=(1234) is still referenced from table "resource_entity".\n', 'DELETE FROM resource_foo', {})
        self.assertEqual('resource_foo', matched.table)
        self.assertEqual('foo_fkey', matched.constraint)
        self.assertEqual('id', matched.key)
        self.assertEqual('resource_entity', matched.key_table)