from unittest import mock
from urllib import parse
import fixtures
import sqlalchemy
from sqlalchemy import Boolean, Index, Integer, DateTime, String
from sqlalchemy import MetaData, Table, Column
from sqlalchemy import ForeignKey, ForeignKeyConstraint
from sqlalchemy.dialects.postgresql import psycopg2
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import column_property
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import registry
from sqlalchemy.orm import Session
from sqlalchemy import sql
from sqlalchemy.sql.expression import cast
from sqlalchemy.sql import select
from sqlalchemy.types import UserDefinedType
from oslo_db import exception
from oslo_db.sqlalchemy import models
from oslo_db.sqlalchemy import provision
from oslo_db.sqlalchemy import session
from oslo_db.sqlalchemy import utils
from oslo_db.tests import base as test_base
from oslo_db.tests.sqlalchemy import base as db_test_base
class TestConnectionUtils(test_base.BaseTestCase):

    def setUp(self):
        super(TestConnectionUtils, self).setUp()
        self.full_credentials = {'backend': 'postgresql+psycopg2', 'database': 'test', 'user': 'dude', 'passwd': 'pass'}
        self.connect_string = 'postgresql+psycopg2://dude:pass@localhost/test'

        @classmethod
        def fake_dbapi(cls):
            return mock.MagicMock()

        class OurDialect(psycopg2.PGDialect_psycopg2):

            def dbapi(self):
                return fake_dbapi

            def import_dbapi(self):
                return fake_dbapi
        patch_dbapi = mock.patch.object(psycopg2, 'PGDialect_psycopg2', new=OurDialect)
        patch_dbapi.start()
        self.addCleanup(patch_dbapi.stop)
        patch_onconnect = mock.patch.object(psycopg2.PGDialect_psycopg2, 'on_connect')
        patch_onconnect.start()
        self.addCleanup(patch_onconnect.stop)

    def test_ensure_backend_available(self):
        with mock.patch.object(sqlalchemy.engine.base.Engine, 'connect') as mock_connect:
            fake_connection = mock.Mock()
            mock_connect.return_value = fake_connection
            eng = provision.Backend._ensure_backend_available(self.connect_string)
            self.assertIsInstance(eng, sqlalchemy.engine.base.Engine)
            self.assertEqual(utils.make_url(self.connect_string), eng.url)
            mock_connect.assert_called_once()
            fake_connection.close.assert_called_once()

    def test_ensure_backend_available_no_connection_raises(self):
        log = self.useFixture(fixtures.FakeLogger())
        err = OperationalError("Can't connect to database", None, None)
        with mock.patch.object(sqlalchemy.engine.base.Engine, 'connect') as mock_connect:
            mock_connect.side_effect = err
            exc = self.assertRaises(exception.BackendNotAvailable, provision.Backend._ensure_backend_available, self.connect_string)
            self.assertEqual("Backend 'postgresql+psycopg2' is unavailable: Could not connect", str(exc))
            self.assertEqual('The postgresql+psycopg2 backend is unavailable: %s' % err, log.output.strip())

    def test_ensure_backend_available_no_dbapi_raises(self):
        log = self.useFixture(fixtures.FakeLogger())
        with mock.patch.object(sqlalchemy, 'create_engine') as mock_create:
            mock_create.side_effect = ImportError("Can't import DBAPI module foobar")
            exc = self.assertRaises(exception.BackendNotAvailable, provision.Backend._ensure_backend_available, self.connect_string)
            mock_create.assert_called_once_with(utils.make_url(self.connect_string))
            self.assertEqual("Backend 'postgresql+psycopg2' is unavailable: No DBAPI installed", str(exc))
            self.assertEqual("The postgresql+psycopg2 backend is unavailable: Can't import DBAPI module foobar", log.output.strip())

    def test_get_db_connection_info(self):
        conn_pieces = parse.urlparse(self.connect_string)
        self.assertEqual(('dude', 'pass', 'test', 'localhost'), utils.get_db_connection_info(conn_pieces))