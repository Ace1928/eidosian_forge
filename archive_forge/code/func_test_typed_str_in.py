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
def test_typed_str_in(self):
    """test related to #7292.

        as a type is given to the bound param, there is no ambiguity
        to the type of element.

        """
    stmt = text('select id FROM some_table WHERE z IN :q ORDER BY id').bindparams(bindparam('q', type_=String, expanding=True))
    self._assert_result(stmt, [(2,), (3,), (4,)], params={'q': ['z2', 'z3', 'z4']})