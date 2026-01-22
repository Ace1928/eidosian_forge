from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from functools import wraps
import re
from . import dictionary
from .types import _OracleBoolean
from .types import _OracleDate
from .types import BFILE
from .types import BINARY_DOUBLE
from .types import BINARY_FLOAT
from .types import DATE
from .types import FLOAT
from .types import INTERVAL
from .types import LONG
from .types import NCLOB
from .types import NUMBER
from .types import NVARCHAR2  # noqa
from .types import OracleRaw  # noqa
from .types import RAW
from .types import ROWID  # noqa
from .types import TIMESTAMP
from .types import VARCHAR2  # noqa
from ... import Computed
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import util
from ...engine import default
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import and_
from ...sql import bindparam
from ...sql import compiler
from ...sql import expression
from ...sql import func
from ...sql import null
from ...sql import or_
from ...sql import select
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql import visitors
from ...sql.visitors import InternalTraversal
from ...types import BLOB
from ...types import CHAR
from ...types import CLOB
from ...types import DOUBLE_PRECISION
from ...types import INTEGER
from ...types import NCHAR
from ...types import NVARCHAR
from ...types import REAL
from ...types import VARCHAR
def _get_nonansi_join_whereclause(self, froms):
    clauses = []

    def visit_join(join):
        if join.isouter:

            def visit_binary(binary):
                if isinstance(binary.left, expression.ColumnClause) and join.right.is_derived_from(binary.left.table):
                    binary.left = _OuterJoinColumn(binary.left)
                elif isinstance(binary.right, expression.ColumnClause) and join.right.is_derived_from(binary.right.table):
                    binary.right = _OuterJoinColumn(binary.right)
            clauses.append(visitors.cloned_traverse(join.onclause, {}, {'binary': visit_binary}))
        else:
            clauses.append(join.onclause)
        for j in (join.left, join.right):
            if isinstance(j, expression.Join):
                visit_join(j)
            elif isinstance(j, expression.FromGrouping):
                visit_join(j.element)
    for f in froms:
        if isinstance(f, expression.Join):
            visit_join(f)
    if not clauses:
        return None
    else:
        return sql.and_(*clauses)