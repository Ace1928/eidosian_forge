import collections.abc as collections_abc
import itertools
from .. import AssertsCompiledSQL
from .. import AssertsExecutionResults
from .. import config
from .. import fixtures
from ..assertions import assert_raises
from ..assertions import eq_
from ..assertions import in_
from ..assertsql import CursorSQL
from ..schema import Column
from ..schema import Table
from ... import bindparam
from ... import case
from ... import column
from ... import Computed
from ... import exists
from ... import false
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import Integer
from ... import literal
from ... import literal_column
from ... import null
from ... import select
from ... import String
from ... import table
from ... import testing
from ... import text
from ... import true
from ... import tuple_
from ... import TupleType
from ... import union
from ... import values
from ...exc import DatabaseError
from ...exc import ProgrammingError
class OrderByLabelTest(fixtures.TablesTest):
    """Test the dialect sends appropriate ORDER BY expressions when
    labels are used.

    This essentially exercises the "supports_simple_order_by_label"
    setting.

    """
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('x', Integer), Column('y', Integer), Column('q', String(50)), Column('p', String(50)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'x': 1, 'y': 2, 'q': 'q1', 'p': 'p3'}, {'id': 2, 'x': 2, 'y': 3, 'q': 'q2', 'p': 'p2'}, {'id': 3, 'x': 3, 'y': 4, 'q': 'q3', 'p': 'p1'}])

    def _assert_result(self, select, result):
        with config.db.connect() as conn:
            eq_(conn.execute(select).fetchall(), result)

    def test_plain(self):
        table = self.tables.some_table
        lx = table.c.x.label('lx')
        self._assert_result(select(lx).order_by(lx), [(1,), (2,), (3,)])

    def test_composed_int(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label('lx')
        self._assert_result(select(lx).order_by(lx), [(3,), (5,), (7,)])

    def test_composed_multiple(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label('lx')
        ly = (func.lower(table.c.q) + table.c.p).label('ly')
        self._assert_result(select(lx, ly).order_by(lx, ly.desc()), [(3, 'q1p3'), (5, 'q2p2'), (7, 'q3p1')])

    def test_plain_desc(self):
        table = self.tables.some_table
        lx = table.c.x.label('lx')
        self._assert_result(select(lx).order_by(lx.desc()), [(3,), (2,), (1,)])

    def test_composed_int_desc(self):
        table = self.tables.some_table
        lx = (table.c.x + table.c.y).label('lx')
        self._assert_result(select(lx).order_by(lx.desc()), [(7,), (5,), (3,)])

    @testing.requires.group_by_complex_expression
    def test_group_by_composed(self):
        table = self.tables.some_table
        expr = (table.c.x + table.c.y).label('lx')
        stmt = select(func.count(table.c.id), expr).group_by(expr).order_by(expr)
        self._assert_result(stmt, [(1, 3), (1, 5), (1, 7)])