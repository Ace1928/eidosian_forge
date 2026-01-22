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
class TestGetUniqueKeys(test_base.BaseTestCase):

    def test_multiple_primary_keys(self):
        self.assertEqual([{'key1', 'key2'}], utils.get_unique_keys(FakeTableWithMultipleKeys))

    def test_unique_index(self):
        self.assertEqual([{'id'}, {'key1', 'key2'}], utils.get_unique_keys(FakeTableWithIndexes))

    def test_unknown_primary_keys(self):
        self.assertIsNone(utils.get_unique_keys(object))

    def test_cache(self):

        class CacheTable(object):
            info = {}
            constraints_called = 0
            indexes_called = 0

            @property
            def constraints(self):
                self.constraints_called += 1
                return []

            @property
            def indexes(self):
                self.indexes_called += 1
                return []

        class CacheModel(object):
            pass
        table = CacheTable()
        mapper_mock = mock.Mock(mapped_table=table, local_table=table)
        mapper_mock.base_mapper = mapper_mock
        mock_inspect = mock.Mock(return_value=mapper_mock)
        model = CacheModel()
        self.assertNotIn('oslodb_unique_keys', CacheTable.info)
        with mock.patch('oslo_db.sqlalchemy.utils.inspect', mock_inspect):
            utils.get_unique_keys(model)
        self.assertIn('oslodb_unique_keys', CacheTable.info)
        self.assertEqual(1, table.constraints_called)
        self.assertEqual(1, table.indexes_called)
        for i in range(10):
            utils.get_unique_keys(model)
        self.assertEqual(1, table.constraints_called)
        self.assertEqual(1, table.indexes_called)