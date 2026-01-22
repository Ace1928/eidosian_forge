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
class CollateTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table('some_table', metadata, Column('id', Integer, primary_key=True), Column('data', String(100)))

    @classmethod
    def insert_data(cls, connection):
        connection.execute(cls.tables.some_table.insert(), [{'id': 1, 'data': 'collate data1'}, {'id': 2, 'data': 'collate data2'}])

    def _assert_result(self, select, result):
        with config.db.connect() as conn:
            eq_(conn.execute(select).fetchall(), result)

    @testing.requires.order_by_collation
    def test_collate_order_by(self):
        collation = testing.requires.get_order_by_collation(testing.config)
        self._assert_result(select(self.tables.some_table).order_by(self.tables.some_table.c.data.collate(collation).asc()), [(1, 'collate data1'), (2, 'collate data2')])