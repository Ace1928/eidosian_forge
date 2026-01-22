from __future__ import annotations
import itertools
import typing as t
from sqlglot import exp
from sqlglot.helper import is_date_unit, is_iso_date, is_iso_datetime
def add_text_to_concat(node: exp.Expression) -> exp.Expression:
    if isinstance(node, exp.Add) and node.type and (node.type.this in exp.DataType.TEXT_TYPES):
        node = exp.Concat(expressions=[node.left, node.right])
    return node