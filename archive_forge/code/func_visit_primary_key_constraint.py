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
def visit_primary_key_constraint(self, constraint, **kw):
    if len(constraint.columns) == 1:
        c = list(constraint)[0]
        if c.primary_key and c.table.dialect_options['sqlite']['autoincrement'] and issubclass(c.type._type_affinity, sqltypes.Integer) and (not c.foreign_keys):
            return None
    text = super().visit_primary_key_constraint(constraint)
    on_conflict_clause = constraint.dialect_options['sqlite']['on_conflict']
    if on_conflict_clause is None and len(constraint.columns) == 1:
        on_conflict_clause = list(constraint)[0].dialect_options['sqlite']['on_conflict_primary_key']
    if on_conflict_clause is not None:
        text += ' ON CONFLICT ' + on_conflict_clause
    return text