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
@reflection.flexi_cache(('schema', InternalTraversal.dp_string), ('dblink', InternalTraversal.dp_string), ('all_objects', InternalTraversal.dp_string_list))
def _get_indexes_rows(self, connection, schema, dblink, all_objects, **kw):
    owner = self.denormalize_schema_name(schema or self.default_schema_name)
    query = self._index_query(owner)
    pks = {row_dict['constraint_name'] for row_dict in self._get_all_constraint_rows(connection, schema, dblink, all_objects, **kw) if row_dict['constraint_type'] == 'P'}
    result = self._run_batches(connection, query, dblink, returns_long=True, mappings=True, all_objects=all_objects)
    return [row_dict for row_dict in result if row_dict['index_name'] not in pks]