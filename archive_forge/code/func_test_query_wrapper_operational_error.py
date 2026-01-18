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
def test_query_wrapper_operational_error(self):
    """test an operational error from query.all() raised as-is."""
    _session = self.sessionmaker()
    _session.begin()
    self.addCleanup(_session.rollback)
    q = _session.query(self.Foo).filter(self.Foo.counter == sqla.func.imfake(123))
    matched = self.assertRaises(sqla.exc.OperationalError, q.all)
    self.assertIn('no such function', str(matched))