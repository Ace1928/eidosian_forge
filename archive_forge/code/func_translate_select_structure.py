from __future__ import annotations
import codecs
import datetime
import operator
import re
from typing import overload
from typing import TYPE_CHECKING
from uuid import UUID as _python_UUID
from . import information_schema as ischema
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import Identity
from ... import schema as sa_schema
from ... import Sequence
from ... import sql
from ... import text
from ... import util
from ...engine import cursor as _cursor
from ...engine import default
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import func
from ...sql import quoted_name
from ...sql import roles
from ...sql import sqltypes
from ...sql import try_cast as try_cast  # noqa: F401
from ...sql import util as sql_util
from ...sql._typing import is_sql_compiler
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.elements import TryCast as TryCast  # noqa: F401
from ...types import BIGINT
from ...types import BINARY
from ...types import CHAR
from ...types import DATE
from ...types import DATETIME
from ...types import DECIMAL
from ...types import FLOAT
from ...types import INTEGER
from ...types import NCHAR
from ...types import NUMERIC
from ...types import NVARCHAR
from ...types import SMALLINT
from ...types import TEXT
from ...types import VARCHAR
from ...util import update_wrapper
from ...util.typing import Literal
from
from
def translate_select_structure(self, select_stmt, **kwargs):
    """Look for ``LIMIT`` and OFFSET in a select statement, and if
        so tries to wrap it in a subquery with ``row_number()`` criterion.
        MSSQL 2012 and above are excluded

        """
    select = select_stmt
    if select._has_row_limiting_clause and (not self.dialect._supports_offset_fetch) and (not self._use_top(select)) and (not getattr(select, '_mssql_visit', None)):
        self._check_can_use_fetch_limit(select)
        _order_by_clauses = [sql_util.unwrap_label_reference(elem) for elem in select._order_by_clause.clauses]
        limit_clause = self._get_limit_or_fetch(select)
        offset_clause = select._offset_clause
        select = select._generate()
        select._mssql_visit = True
        select = select.add_columns(sql.func.ROW_NUMBER().over(order_by=_order_by_clauses).label('mssql_rn')).order_by(None).alias()
        mssql_rn = sql.column('mssql_rn')
        limitselect = sql.select(*[c for c in select.c if c.key != 'mssql_rn'])
        if offset_clause is not None:
            limitselect = limitselect.where(mssql_rn > offset_clause)
            if limit_clause is not None:
                limitselect = limitselect.where(mssql_rn <= limit_clause + offset_clause)
        else:
            limitselect = limitselect.where(mssql_rn <= limit_clause)
        return limitselect
    else:
        return select