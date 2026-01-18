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
def test_operational_error_asis(self):
    """Test operational errors.

        test that SQLAlchemy OperationalErrors that aren't disconnects
        are passed through without wrapping.
        """
    matched = self._run_test('mysql', 'select some_operational_error', self.OperationalError('some op error'), sqla.exc.OperationalError)
    self.assertSQLAException(matched, 'OperationalError', 'some op error')