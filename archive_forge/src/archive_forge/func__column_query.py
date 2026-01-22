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
def _column_query(self, owner):
    all_cols = dictionary.all_tab_cols
    all_comments = dictionary.all_col_comments
    all_ids = dictionary.all_tab_identity_cols
    if self.server_version_info >= (12,):
        add_cols = (all_cols.c.default_on_null, sql.case((all_ids.c.table_name.is_(None), sql.null()), else_=all_ids.c.generation_type + ',' + all_ids.c.identity_options).label('identity_options'))
        join_identity_cols = True
    else:
        add_cols = (sql.null().label('default_on_null'), sql.null().label('identity_options'))
        join_identity_cols = False
    query = select(all_cols.c.table_name, all_cols.c.column_name, all_cols.c.data_type, all_cols.c.char_length, all_cols.c.data_precision, all_cols.c.data_scale, all_cols.c.nullable, all_cols.c.data_default, all_comments.c.comments, all_cols.c.virtual_column, *add_cols).select_from(all_cols).outerjoin(all_comments, and_(all_cols.c.table_name == all_comments.c.table_name, all_cols.c.column_name == all_comments.c.column_name, all_cols.c.owner == all_comments.c.owner))
    if join_identity_cols:
        query = query.outerjoin(all_ids, and_(all_cols.c.table_name == all_ids.c.table_name, all_cols.c.column_name == all_ids.c.column_name, all_cols.c.owner == all_ids.c.owner))
    query = query.where(all_cols.c.table_name.in_(bindparam('all_objects')), all_cols.c.hidden_column == 'NO', all_cols.c.owner == owner).order_by(all_cols.c.table_name, all_cols.c.column_id)
    return query