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
def _get_table_pragma(self, connection, pragma, table_name, schema=None):
    quote = self.identifier_preparer.quote_identifier
    if schema is not None:
        statements = [f'PRAGMA {quote(schema)}.']
    else:
        statements = ['PRAGMA main.', 'PRAGMA temp.']
    qtable = quote(table_name)
    for statement in statements:
        statement = f'{statement}{pragma}({qtable})'
        cursor = connection.exec_driver_sql(statement)
        if not cursor._soft_closed:
            result = cursor.fetchall()
        else:
            result = []
        if result:
            return result
    else:
        return []