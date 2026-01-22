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
@lru_cache()
def _table_options_query(self, owner, scope, kind, has_filter_names, has_mat_views):
    query = select(dictionary.all_tables.c.table_name, dictionary.all_tables.c.compression, dictionary.all_tables.c.compress_for).where(dictionary.all_tables.c.owner == owner)
    if has_filter_names:
        query = query.where(dictionary.all_tables.c.table_name.in_(bindparam('filter_names')))
    if scope is ObjectScope.DEFAULT:
        query = query.where(dictionary.all_tables.c.duration.is_(null()))
    elif scope is ObjectScope.TEMPORARY:
        query = query.where(dictionary.all_tables.c.duration.is_not(null()))
    if has_mat_views and ObjectKind.TABLE in kind and (ObjectKind.MATERIALIZED_VIEW not in kind):
        query = query.where(dictionary.all_tables.c.table_name.not_in(bindparam('mat_views')))
    elif ObjectKind.TABLE not in kind and ObjectKind.MATERIALIZED_VIEW in kind:
        query = query.where(dictionary.all_tables.c.table_name.in_(bindparam('mat_views')))
    return query