from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def _coerce_datediff_args(node: exp.DateDiff) -> None:
    for e in (node.this, node.expression):
        if e.type.this not in exp.DataType.TEMPORAL_TYPES:
            e.replace(exp.cast(e.copy(), to=exp.DataType.Type.DATETIME))