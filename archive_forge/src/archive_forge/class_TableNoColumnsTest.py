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
class TableNoColumnsTest(fixtures.TestBase):
    __requires__ = ('reflect_tables_no_columns',)
    __backend__ = True

    @testing.fixture
    def table_no_columns(self, connection, metadata):
        Table('empty', metadata)
        metadata.create_all(connection)

    @testing.fixture
    def view_no_columns(self, connection, metadata):
        Table('empty', metadata)
        event.listen(metadata, 'after_create', DDL('CREATE VIEW empty_v AS SELECT * FROM empty'))
        event.listen(metadata, 'before_drop', DDL('DROP VIEW IF EXISTS empty_v'))
        metadata.create_all(connection)

    def test_reflect_table_no_columns(self, connection, table_no_columns):
        t2 = Table('empty', MetaData(), autoload_with=connection)
        eq_(list(t2.c), [])

    def test_get_columns_table_no_columns(self, connection, table_no_columns):
        insp = inspect(connection)
        eq_(insp.get_columns('empty'), [])
        multi = insp.get_multi_columns()
        eq_(multi, {(None, 'empty'): []})

    def test_reflect_incl_table_no_columns(self, connection, table_no_columns):
        m = MetaData()
        m.reflect(connection)
        assert set(m.tables).intersection(['empty'])

    @testing.requires.views
    def test_reflect_view_no_columns(self, connection, view_no_columns):
        t2 = Table('empty_v', MetaData(), autoload_with=connection)
        eq_(list(t2.c), [])

    @testing.requires.views
    def test_get_columns_view_no_columns(self, connection, view_no_columns):
        insp = inspect(connection)
        eq_(insp.get_columns('empty_v'), [])
        multi = insp.get_multi_columns(kind=ObjectKind.VIEW)
        eq_(multi, {(None, 'empty_v'): []})