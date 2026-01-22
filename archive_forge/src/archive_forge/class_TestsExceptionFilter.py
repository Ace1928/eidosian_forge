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
class TestsExceptionFilter(_SQLAExceptionMatcher, test_base.BaseTestCase):

    class Error(Exception):
        """DBAPI base error.

        This exception and subclasses are used in a mock context
        within these tests.

        """

    class DataError(Error):
        pass

    class OperationalError(Error):
        pass

    class InterfaceError(Error):
        pass

    class InternalError(Error):
        pass

    class IntegrityError(Error):
        pass

    class ProgrammingError(Error):
        pass

    class TransactionRollbackError(OperationalError):
        """Special psycopg2-only error class.

        SQLAlchemy has an issue with this per issue #3075:

        https://bitbucket.org/zzzeek/sqlalchemy/issue/3075/

        """

    def setUp(self):
        super(TestsExceptionFilter, self).setUp()
        self.engine = sqla.create_engine('sqlite://')
        exc_filters.register_engine(self.engine)
        self.engine.connect().close()

    @contextlib.contextmanager
    def _dbapi_fixture(self, dialect_name, is_disconnect=False):
        engine = self.engine
        with test_utils.nested(mock.patch.object(engine.dialect.dbapi, 'Error', self.Error), mock.patch.object(engine.dialect, 'name', dialect_name), mock.patch.object(engine.dialect, 'is_disconnect', lambda *args: is_disconnect)):
            yield

    @contextlib.contextmanager
    def _fixture(self, dialect_name, exception, is_disconnect=False):

        def do_execute(self, cursor, statement, parameters, **kw):
            raise exception
        engine = self.engine
        self.engine.connect().close()
        patches = [mock.patch.object(engine.dialect, 'do_execute', do_execute), mock.patch.object(engine.dialect, 'dbapi', mock.Mock(Error=self.Error)), mock.patch.object(engine.dialect, 'name', dialect_name), mock.patch.object(engine.dialect, 'is_disconnect', lambda *args: is_disconnect)]
        if compat.sqla_2:
            patches.append(mock.patch.object(engine.dialect, 'loaded_dbapi', mock.Mock(Error=self.Error)))
        with test_utils.nested(*patches):
            yield

    def _run_test(self, dialect_name, statement, raises, expected, is_disconnect=False, params=()):
        with self._fixture(dialect_name, raises, is_disconnect=is_disconnect):
            with self.engine.connect() as conn:
                matched = self.assertRaises(expected, conn.execute, sql.text(statement), params)
                return matched