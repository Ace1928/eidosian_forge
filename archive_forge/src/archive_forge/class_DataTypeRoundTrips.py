import datetime
from unittest import mock
import uuid
import fixtures
import freezegun
from oslo_db import exception as db_exception
from oslo_db import options
from oslo_log import log
import sqlalchemy
from sqlalchemy import exc
from testtools import matchers
from keystone.common import driver_hints
from keystone.common import provider_api
from keystone.common import sql
from keystone.common.sql import core
import keystone.conf
from keystone.credential.providers import fernet as credential_provider
from keystone import exception
from keystone.identity.backends import sql_model as identity_sql
from keystone.resource.backends import base as resource
from keystone.tests import unit
from keystone.tests.unit.assignment import test_backends as assignment_tests
from keystone.tests.unit.catalog import test_backends as catalog_tests
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.identity import test_backends as identity_tests
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit.limit import test_backends as limit_tests
from keystone.tests.unit.policy import test_backends as policy_tests
from keystone.tests.unit.resource import test_backends as resource_tests
from keystone.tests.unit.trust import test_backends as trust_tests
from keystone.trust.backends import sql as trust_sql
class DataTypeRoundTrips(SqlTests):

    def test_json_blob_roundtrip(self):
        """Test round-trip of a JSON data structure with JsonBlob."""
        with sql.session_for_read() as session:
            val = session.scalar(sqlalchemy.select(sqlalchemy.literal({'key': 'value'}, type_=core.JsonBlob)))
        self.assertEqual({'key': 'value'}, val)

    def test_json_blob_sql_null(self):
        """Test that JsonBlob can accommodate a SQL NULL value in a result set.

        SQL NULL may be handled by JsonBlob in the case where a table is
        storing NULL in a JsonBlob column, as several models use this type
        in a column that is nullable.   It also comes back when the column
        is left NULL from being in an OUTER JOIN.  In Python, this means
        the None constant is handled by the datatype.

        """
        with sql.session_for_read() as session:
            val = session.scalar(sqlalchemy.select(sqlalchemy.cast(sqlalchemy.null(), type_=core.JsonBlob)))
        self.assertIsNone(val)

    def test_json_blob_python_none(self):
        """Test that JsonBlob round-trips a Python None.

        This is where JSON datatypes get a little nutty, in that JSON has
        a 'null' keyword, and JsonBlob right now will persist Python None
        as the json string 'null', not SQL NULL.

        """
        with sql.session_for_read() as session:
            val = session.scalar(sqlalchemy.select(sqlalchemy.literal(None, type_=core.JsonBlob)))
        self.assertIsNone(val)

    def test_json_blob_python_none_renders(self):
        """Test that JsonBlob actually renders JSON 'null' for Python None."""
        with sql.session_for_read() as session:
            val = session.scalar(sqlalchemy.select(sqlalchemy.cast(sqlalchemy.literal(None, type_=core.JsonBlob), sqlalchemy.String)))
        self.assertEqual('null', val)

    def test_datetimeint_roundtrip(self):
        """Test round-trip of a Python datetime with DateTimeInt."""
        with sql.session_for_read() as session:
            datetime_value = datetime.datetime(2019, 5, 15, 10, 17, 55)
            val = session.scalar(sqlalchemy.select(sqlalchemy.literal(datetime_value, type_=core.DateTimeInt)))
        self.assertEqual(datetime_value, val)

    def test_datetimeint_persistence(self):
        """Test integer persistence with DateTimeInt."""
        with sql.session_for_read() as session:
            datetime_value = datetime.datetime(2019, 5, 15, 10, 17, 55)
            val = session.scalar(sqlalchemy.select(sqlalchemy.cast(sqlalchemy.literal(datetime_value, type_=core.DateTimeInt), sqlalchemy.Integer)))
        self.assertEqual(1557915475000000, val)

    def test_datetimeint_python_none(self):
        """Test round-trip of a Python None with DateTimeInt."""
        with sql.session_for_read() as session:
            val = session.scalar(sqlalchemy.select(sqlalchemy.literal(None, type_=core.DateTimeInt)))
        self.assertIsNone(val)