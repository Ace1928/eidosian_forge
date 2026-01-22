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
def _all_objects_query(self, owner, scope, kind, has_filter_names, has_mat_views):
    query = select(dictionary.all_objects.c.object_name).select_from(dictionary.all_objects).where(dictionary.all_objects.c.owner == owner)
    if kind is ObjectKind.ANY:
        query = query.where(dictionary.all_objects.c.object_type.in_(('TABLE', 'VIEW')))
    else:
        object_type = []
        if ObjectKind.VIEW in kind:
            object_type.append('VIEW')
        if ObjectKind.MATERIALIZED_VIEW in kind and ObjectKind.TABLE not in kind:
            object_type.append('MATERIALIZED VIEW')
        if ObjectKind.TABLE in kind:
            object_type.append('TABLE')
            if has_mat_views and ObjectKind.MATERIALIZED_VIEW not in kind:
                query = query.where(dictionary.all_objects.c.object_name.not_in(bindparam('mat_views')))
        query = query.where(dictionary.all_objects.c.object_type.in_(object_type))
    if scope is ObjectScope.DEFAULT:
        query = query.where(dictionary.all_objects.c.temporary == 'N')
    elif scope is ObjectScope.TEMPORARY:
        query = query.where(dictionary.all_objects.c.temporary == 'Y')
    if has_filter_names:
        query = query.where(dictionary.all_objects.c.object_name.in_(bindparam('filter_names')))
    return query