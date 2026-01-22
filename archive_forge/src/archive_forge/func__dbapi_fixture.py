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
@contextlib.contextmanager
def _dbapi_fixture(self, dialect_name, is_disconnect=False):
    engine = self.engine
    with test_utils.nested(mock.patch.object(engine.dialect.dbapi, 'Error', self.Error), mock.patch.object(engine.dialect, 'name', dialect_name), mock.patch.object(engine.dialect, 'is_disconnect', lambda *args: is_disconnect)):
        yield