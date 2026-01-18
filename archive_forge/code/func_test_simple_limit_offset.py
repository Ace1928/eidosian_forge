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
@testing.combinations([(2, 0), (2, 1), (3, 2)], [(2, 1), (2, 0), (3, 2)], [(3, 1), (2, 1), (3, 1)], argnames='cases')
@testing.requires.offset
def test_simple_limit_offset(self, connection, cases):
    table = self.tables.some_table
    connection = connection.execution_options(compiled_cache={})
    assert_data = [(1, 1, 2), (2, 2, 3), (3, 3, 4), (4, 4, 5), (5, 4, 6)]
    for limit, offset in cases:
        expected = assert_data[offset:offset + limit]
        self._assert_result(connection, select(table).order_by(table.c.id).limit(limit).offset(offset), expected)