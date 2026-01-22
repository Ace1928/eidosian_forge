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
class FetchLimitOffsetTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('x', Integer), Column('y', Integer))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'x': 1, 'y': 2}, {'id': 2, 'x': 2, 'y': 3}, {'id': 3, 'x': 3, 'y': 4}, {'id': 4, 'x': 4, 'y': 5}, {'id': 5, 'x': 4, 'y': 6}])

    def _assert_result(self, connection, select, result, params=(), set_=False):
        if set_:
            query_res = connection.execute(select, params).fetchall()
            eq_(len(query_res), len(result))
            eq_(set(query_res), set(result))
        else:
            eq_(connection.execute(select, params).fetchall(), result)

    def _assert_result_str(self, select, result, params=()):
        with config.db.connect() as conn:
            eq_(conn.exec_driver_sql(select, params).fetchall(), result)

    def test_simple_limit(self, connection):
        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id)
        self._assert_result(connection, stmt.limit(2), [(1, 1, 2), (2, 2, 3)])
        self._assert_result(connection, stmt.limit(3), [(1, 1, 2), (2, 2, 3), (3, 3, 4)])

    def test_limit_render_multiple_times(self, connection):
        table = self.tables.some_table
        stmt = select(table.c.id).limit(1).scalar_subquery()
        u = union(select(stmt), select(stmt)).subquery().select()
        self._assert_result(connection, u, [(1,)])

    @testing.requires.fetch_first
    def test_simple_fetch(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(2), [(1, 1, 2), (2, 2, 3)])
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(3), [(1, 1, 2), (2, 2, 3), (3, 3, 4)])

    @testing.requires.offset
    def test_simple_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).offset(2), [(3, 3, 4), (4, 4, 5), (5, 4, 6)])
        self._assert_result(connection, select(table).order_by(table.c.id).offset(3), [(4, 4, 5), (5, 4, 6)])

    @testing.combinations([(2, 0), (2, 1), (3, 2)], [(2, 1), (2, 0), (3, 2)], [(3, 1), (2, 1), (3, 1)], argnames='cases')
    @testing.requires.offset
    def test_simple_limit_offset(self, connection, cases):
        table = self.tables.some_table
        connection = connection.execution_options(compiled_cache={})
        assert_data = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)]
        for limit, offset in cases:
            expected = assert_data[offset:offset + limit]
            self._assert_result(connection, select(table).order_by(table.c.id).limit(limit).offset(offset), expected)

    @testing.requires.fetch_first
    def test_simple_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(2).offset(1), [(2, 2, 3), (3, 3, 4)])
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(3).offset(2), [(3, 3, 4), (4, 4, 5), (5, 4, 6)])

    @testing.requires.fetch_no_order_by
    def test_fetch_offset_no_order(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).fetch(10), [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)], set_=True)

    @testing.requires.offset
    def test_simple_offset_zero(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).offset(0), [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)])
        self._assert_result(connection, select(table).order_by(table.c.id).offset(1), [(2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)])

    @testing.requires.offset
    def test_limit_offset_nobinds(self):
        """test that 'literal binds' mode works - no bound params."""
        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id).limit(2).offset(1)
        sql = stmt.compile(dialect=config.db.dialect, compile_kwargs={'literal_binds': True})
        sql = str(sql)
        self._assert_result_str(sql, [(2, 2, 3), (3, 3, 4)])

    @testing.requires.fetch_first
    def test_fetch_offset_nobinds(self):
        """test that 'literal binds' mode works - no bound params."""
        table = self.tables.some_table
        stmt = select(table).order_by(table.c.id).fetch(2).offset(1)
        sql = stmt.compile(dialect=config.db.dialect, compile_kwargs={'literal_binds': True})
        sql = str(sql)
        self._assert_result_str(sql, [(2, 2, 3), (3, 3, 4)])

    @testing.requires.bound_limit_offset
    def test_bound_limit(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(bindparam('l')), [(1, 1, 2), (2, 2, 3)], params={'l': 2})
        self._assert_result(connection, select(table).order_by(table.c.id).limit(bindparam('l')), [(1, 1, 2), (2, 2, 3), (3, 3, 4)], params={'l': 3})

    @testing.requires.bound_limit_offset
    def test_bound_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).offset(bindparam('o')), [(3, 3, 4), (4, 4, 5), (5, 4, 6)], params={'o': 2})
        self._assert_result(connection, select(table).order_by(table.c.id).offset(bindparam('o')), [(2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)], params={'o': 1})

    @testing.requires.bound_limit_offset
    def test_bound_limit_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(bindparam('l')).offset(bindparam('o')), [(2, 2, 3), (3, 3, 4)], params={'l': 2, 'o': 1})
        self._assert_result(connection, select(table).order_by(table.c.id).limit(bindparam('l')).offset(bindparam('o')), [(3, 3, 4), (4, 4, 5), (5, 4, 6)], params={'l': 3, 'o': 2})

    @testing.requires.fetch_first
    def test_bound_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(bindparam('f')).offset(bindparam('o')), [(2, 2, 3), (3, 3, 4)], params={'f': 2, 'o': 1})
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(bindparam('f')).offset(bindparam('o')), [(3, 3, 4), (4, 4, 5), (5, 4, 6)], params={'f': 3, 'o': 2})

    @testing.requires.sql_expression_limit_offset
    def test_expr_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).offset(literal_column('1') + literal_column('2')), [(4, 4, 5), (5, 4, 6)])

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(literal_column('1') + literal_column('2')), [(1, 1, 2), (2, 2, 3), (3, 3, 4)])

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(literal_column('1') + literal_column('1')).offset(literal_column('1') + literal_column('1')), [(3, 3, 4), (4, 4, 5)])

    @testing.requires.fetch_first
    @testing.requires.fetch_expression
    def test_expr_fetch_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(literal_column('1') + literal_column('1')).offset(literal_column('1') + literal_column('1')), [(3, 3, 4), (4, 4, 5)])

    @testing.requires.sql_expression_limit_offset
    def test_simple_limit_expr_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(2).offset(literal_column('1') + literal_column('1')), [(3, 3, 4), (4, 4, 5)])
        self._assert_result(connection, select(table).order_by(table.c.id).limit(3).offset(literal_column('1') + literal_column('1')), [(3, 3, 4), (4, 4, 5), (5, 4, 6)])

    @testing.requires.sql_expression_limit_offset
    def test_expr_limit_simple_offset(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).limit(literal_column('1') + literal_column('1')).offset(2), [(3, 3, 4), (4, 4, 5)])
        self._assert_result(connection, select(table).order_by(table.c.id).limit(literal_column('1') + literal_column('1')).offset(1), [(2, 2, 3), (3, 3, 4)])

    @testing.requires.fetch_ties
    def test_simple_fetch_ties(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.x.desc()).fetch(1, with_ties=True), [(4, 4, 5), (5, 4, 6)], set_=True)
        self._assert_result(connection, select(table).order_by(table.c.x.desc()).fetch(3, with_ties=True), [(3, 3, 4), (4, 4, 5), (5, 4, 6)], set_=True)

    @testing.requires.fetch_ties
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_ties(self, connection):
        table = self.tables.some_table
        fa = connection.execute(select(table).order_by(table.c.x).fetch(2, with_ties=True).offset(2)).fetchall()
        eq_(fa[0], (3, 3, 4))
        eq_(set(fa), {(3, 3, 4), (4, 4, 5), (5, 4, 6)})

    @testing.requires.fetch_ties
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_ties_exact_number(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.x).fetch(2, with_ties=True).offset(1), [(2, 2, 3), (3, 3, 4)])
        self._assert_result(connection, select(table).order_by(table.c.x).fetch(3, with_ties=True).offset(3), [(4, 4, 5), (5, 4, 6)])

    @testing.requires.fetch_percent
    def test_simple_fetch_percent(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(20, percent=True), [(1, 1, 2)])

    @testing.requires.fetch_percent
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_percent(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.id).fetch(40, percent=True).offset(1), [(2, 2, 3), (3, 3, 4)])

    @testing.requires.fetch_ties
    @testing.requires.fetch_percent
    def test_simple_fetch_percent_ties(self, connection):
        table = self.tables.some_table
        self._assert_result(connection, select(table).order_by(table.c.x.desc()).fetch(20, percent=True, with_ties=True), [(4, 4, 5), (5, 4, 6)], set_=True)

    @testing.requires.fetch_ties
    @testing.requires.fetch_percent
    @testing.requires.fetch_offset_with_options
    def test_fetch_offset_percent_ties(self, connection):
        table = self.tables.some_table
        fa = connection.execute(select(table).order_by(table.c.x).fetch(40, percent=True, with_ties=True).offset(2)).fetchall()
        eq_(fa[0], (3, 3, 4))
        eq_(set(fa), {(3, 3, 4), (4, 4, 5), (5, 4, 6)})