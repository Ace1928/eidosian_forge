import operator
import re
import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock
class HasTableTest(OneConnectionTablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('test_table', metadata, Column('id', Integer, primary_key=True), Column('data', String(50)))
        if testing.requires.schemas.enabled:
            Table('test_table_s', metadata, Column('id', Integer, primary_key=True), Column('data', String(50)), schema=config.test_schema)
        if testing.requires.view_reflection:
            cls.define_views(metadata)
        if testing.requires.has_temp_table.enabled:
            cls.define_temp_tables(metadata)

    @classmethod
    def define_views(cls, metadata):
        query = 'CREATE VIEW vv AS SELECT id, data FROM test_table'
        event.listen(metadata, 'after_create', DDL(query))
        event.listen(metadata, 'before_drop', DDL('DROP VIEW vv'))
        if testing.requires.schemas.enabled:
            query = 'CREATE VIEW %s.vv AS SELECT id, data FROM %s.test_table_s' % (config.test_schema, config.test_schema)
            event.listen(metadata, 'after_create', DDL(query))
            event.listen(metadata, 'before_drop', DDL('DROP VIEW %s.vv' % config.test_schema))

    @classmethod
    def temp_table_name(cls):
        return get_temp_table_name(config, config.db, f'user_tmp_{config.ident}')

    @classmethod
    def define_temp_tables(cls, metadata):
        kw = temp_table_keyword_args(config, config.db)
        table_name = cls.temp_table_name()
        user_tmp = Table(table_name, metadata, Column('id', sa.INT, primary_key=True), Column('name', sa.VARCHAR(50)), **kw)
        if testing.requires.view_reflection.enabled and testing.requires.temporary_views.enabled:
            event.listen(user_tmp, 'after_create', DDL('create temporary view user_tmp_v as select * from user_tmp_%s' % config.ident))
            event.listen(user_tmp, 'before_drop', DDL('drop view user_tmp_v'))

    def test_has_table(self):
        with config.db.begin() as conn:
            is_true(config.db.dialect.has_table(conn, 'test_table'))
            is_false(config.db.dialect.has_table(conn, 'test_table_s'))
            is_false(config.db.dialect.has_table(conn, 'nonexistent_table'))

    def test_has_table_cache(self, metadata):
        insp = inspect(config.db)
        is_true(insp.has_table('test_table'))
        nt = Table('new_table', metadata, Column('col', Integer))
        is_false(insp.has_table('new_table'))
        nt.create(config.db)
        try:
            is_false(insp.has_table('new_table'))
            insp.clear_cache()
            is_true(insp.has_table('new_table'))
        finally:
            nt.drop(config.db)

    @testing.requires.schemas
    def test_has_table_schema(self):
        with config.db.begin() as conn:
            is_false(config.db.dialect.has_table(conn, 'test_table', schema=config.test_schema))
            is_true(config.db.dialect.has_table(conn, 'test_table_s', schema=config.test_schema))
            is_false(config.db.dialect.has_table(conn, 'nonexistent_table', schema=config.test_schema))

    @testing.requires.schemas
    def test_has_table_nonexistent_schema(self):
        with config.db.begin() as conn:
            is_false(config.db.dialect.has_table(conn, 'test_table', schema='nonexistent_schema'))

    @testing.requires.views
    def test_has_table_view(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table('vv'))

    @testing.requires.has_temp_table
    def test_has_table_temp_table(self, connection):
        insp = inspect(connection)
        temp_table_name = self.temp_table_name()
        is_true(insp.has_table(temp_table_name))

    @testing.requires.has_temp_table
    @testing.requires.view_reflection
    @testing.requires.temporary_views
    def test_has_table_temp_view(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table('user_tmp_v'))

    @testing.requires.views
    @testing.requires.schemas
    def test_has_table_view_schema(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table('vv', config.test_schema))