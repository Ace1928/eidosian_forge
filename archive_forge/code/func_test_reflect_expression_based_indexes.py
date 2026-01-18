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
@testing.requires.indexes_with_expressions
def test_reflect_expression_based_indexes(self, metadata, connection):
    t = Table('t', metadata, Column('x', String(30)), Column('y', String(30)), Column('z', String(30)))
    Index('t_idx', func.lower(t.c.x), t.c.z, func.lower(t.c.y))
    long_str = 'long string ' * 100
    Index('t_idx_long', func.coalesce(t.c.x, long_str))
    Index('t_idx_2', t.c.x)
    metadata.create_all(connection)
    insp = inspect(connection)
    expected = [{'name': 't_idx_2', 'column_names': ['x'], 'unique': False, 'dialect_options': {}}]

    def completeIndex(entry):
        if testing.requires.index_reflects_included_columns.enabled:
            entry['include_columns'] = []
            entry['dialect_options'] = {f'{connection.engine.name}_include': []}
        else:
            entry.setdefault('dialect_options', {})
    completeIndex(expected[0])

    class lower_index_str(str):

        def __eq__(self, other):
            ol = other.lower()
            return 'lower' in ol and ('x' in ol or 'y' in ol)

    class coalesce_index_str(str):

        def __eq__(self, other):
            return 'coalesce' in other.lower() and long_str in other
    if testing.requires.reflect_indexes_with_expressions.enabled:
        expr_index = {'name': 't_idx', 'column_names': [None, 'z', None], 'expressions': [lower_index_str('lower(x)'), 'z', lower_index_str('lower(y)')], 'unique': False}
        completeIndex(expr_index)
        expected.insert(0, expr_index)
        expr_index_long = {'name': 't_idx_long', 'column_names': [None], 'expressions': [coalesce_index_str(f"coalesce(x, '{long_str}')")], 'unique': False}
        completeIndex(expr_index_long)
        expected.append(expr_index_long)
        eq_(insp.get_indexes('t'), expected)
        m2 = MetaData()
        t2 = Table('t', m2, autoload_with=connection)
    else:
        with expect_warnings('Skipped unsupported reflection of expression-based index t_idx'):
            eq_(insp.get_indexes('t'), expected)
            m2 = MetaData()
            t2 = Table('t', m2, autoload_with=connection)
    self.compare_table_index_with_expected(t2, expected, connection.engine.name)