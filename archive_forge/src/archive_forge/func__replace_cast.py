from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def _replace_cast(node: exp.Expression, to: exp.DataType.Type) -> None:
    node.replace(exp.cast(node.copy(), to=to))