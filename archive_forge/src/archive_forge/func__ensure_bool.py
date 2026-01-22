from __future__ import annotations
import typing as t
from sqlglot import expressions as exp
from sqlglot.helper import find_new_name, name_sequence
def _ensure_bool(node: exp.Expression) -> None:
    if node.is_number or node.is_type(exp.DataType.Type.UNKNOWN, *exp.DataType.NUMERIC_TYPES) or (isinstance(node, exp.Column) and (not node.type)):
        node.replace(node.neq(0))