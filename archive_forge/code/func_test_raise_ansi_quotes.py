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
def test_raise_ansi_quotes(self):
    with self.engine.connect() as conn:
        conn.detach()
        conn.execute(sql.text("SET SESSION sql_mode = 'ANSI';"))
        matched = self.assertRaises(exception.DBReferenceError, conn.execute, self.table_2.insert().values(id=1, foo_id=2))
    self.assertIsInstance(matched.inner_exception, sqlalchemy.exc.IntegrityError)
    self.assertEqual(matched.inner_exception.orig.args[0], 1452)
    self.assertEqual('resource_entity', matched.table)
    self.assertEqual('foo_fkey', matched.constraint)
    self.assertEqual('foo_id', matched.key)
    self.assertEqual('resource_foo', matched.key_table)