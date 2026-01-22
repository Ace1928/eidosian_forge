from __future__ import annotations
import functools
import typing as t
from sqlglot import exp
from sqlglot.helper import (
from sqlglot.optimizer.scope import Scope, traverse_scope
from sqlglot.schema import Schema, ensure_schema
def _coerce_date_literal(l: exp.Expression, unit: t.Optional[exp.Expression]) -> exp.DataType.Type:
    date_text = l.name
    is_iso_date_ = is_iso_date(date_text)
    if is_iso_date_ and is_date_unit(unit):
        return exp.DataType.Type.DATE
    if is_iso_date_ or is_iso_datetime(date_text):
        return exp.DataType.Type.DATETIME
    return exp.DataType.Type.UNKNOWN