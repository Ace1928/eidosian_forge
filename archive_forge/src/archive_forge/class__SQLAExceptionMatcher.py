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
class _SQLAExceptionMatcher(object):

    def assertInnerException(self, matched, exception_type, message, sql=None, params=None):
        exc = matched.inner_exception
        self.assertSQLAException(exc, exception_type, message, sql, params)

    def assertSQLAException(self, exc, exception_type, message, sql=None, params=None):
        if isinstance(exception_type, (type, tuple)):
            self.assertTrue(issubclass(exc.__class__, exception_type))
        else:
            self.assertEqual(exception_type, exc.__class__.__name__)
        if isinstance(message, tuple):
            self.assertEqual([m.lower() if isinstance(m, str) else m for m in message], [a.lower() if isinstance(a, str) else a for a in exc.orig.args])
        else:
            self.assertEqual(message.lower(), str(exc.orig).lower())
        if sql is not None:
            if params is not None:
                if '?' in exc.statement:
                    self.assertEqual(sql, exc.statement)
                    self.assertEqual(params, exc.params)
                else:
                    self.assertEqual(sql % params, exc.statement % exc.params)
            else:
                self.assertEqual(sql, exc.statement)