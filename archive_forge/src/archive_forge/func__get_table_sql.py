from the proposed insertion. These values are specified using the
from __future__ import annotations
import datetime
import numbers
import re
from typing import Optional
from .json import JSON
from .json import JSONIndexType
from .json import JSONPathType
from ... import exc
from ... import schema as sa_schema
from ... import sql
from ... import text
from ... import types as sqltypes
from ... import util
from ...engine import default
from ...engine import processors
from ...engine import reflection
from ...engine.reflection import ReflectionDefaults
from ...sql import coercions
from ...sql import ColumnElement
from ...sql import compiler
from ...sql import elements
from ...sql import roles
from ...sql import schema
from ...types import BLOB  # noqa
from ...types import BOOLEAN  # noqa
from ...types import CHAR  # noqa
from ...types import DECIMAL  # noqa
from ...types import FLOAT  # noqa
from ...types import INTEGER  # noqa
from ...types import NUMERIC  # noqa
from ...types import REAL  # noqa
from ...types import SMALLINT  # noqa
from ...types import TEXT  # noqa
from ...types import TIMESTAMP  # noqa
from ...types import VARCHAR  # noqa
@reflection.cache
def _get_table_sql(self, connection, table_name, schema=None, **kw):
    if schema:
        schema_expr = '%s.' % self.identifier_preparer.quote_identifier(schema)
    else:
        schema_expr = ''
    try:
        s = "SELECT sql FROM  (SELECT * FROM %(schema)ssqlite_master UNION ALL   SELECT * FROM %(schema)ssqlite_temp_master) WHERE name = ? AND type in ('table', 'view')" % {'schema': schema_expr}
        rs = connection.exec_driver_sql(s, (table_name,))
    except exc.DBAPIError:
        s = "SELECT sql FROM %(schema)ssqlite_master WHERE name = ? AND type in ('table', 'view')" % {'schema': schema_expr}
        rs = connection.exec_driver_sql(s, (table_name,))
    value = rs.scalar()
    if value is None and (not self._is_sys_table(table_name)):
        raise exc.NoSuchTableError(f'{schema_expr}{table_name}')
    return value