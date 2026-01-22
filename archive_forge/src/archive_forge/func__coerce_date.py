from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _coerce_date(l: exp.Expression, unit: t.Optional[exp.Expression]) -> exp.DataType.Type:
    if not is_date_unit(unit):
        return exp.DataType.Type.DATETIME
    return l.type.this if l.type else exp.DataType.Type.UNKNOWN