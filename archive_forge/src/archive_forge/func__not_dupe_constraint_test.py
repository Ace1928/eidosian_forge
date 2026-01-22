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
def _not_dupe_constraint_test(self, dialect_name, statement, message, expected_cls):
    matched = self._run_test(dialect_name, statement, self.IntegrityError(message), expected_cls)
    self.assertInnerException(matched, 'IntegrityError', str(self.IntegrityError(message)), statement)