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
class TestDataError(TestsExceptionFilter):

    def _run_bad_data_test(self, dialect_name, message, error_class):
        self._run_test(dialect_name, 'INSERT INTO TABLE some_values', error_class(message), exception.DBDataError)

    def test_bad_data_incorrect_string(self):
        self._run_bad_data_test('mysql', '(1366, "Incorrect string value: \'\\xF0\' for column \'resource\' at row 1"', self.OperationalError)

    def test_bad_data_out_of_range(self):
        self._run_bad_data_test('mysql', '(1264, "Out of range value for column \'resource\' at row 1"', self.DataError)

    def test_data_too_long_for_column(self):
        self._run_bad_data_test('mysql', '(1406, "Data too long for column \'resource\' at row 1"', self.DataError)