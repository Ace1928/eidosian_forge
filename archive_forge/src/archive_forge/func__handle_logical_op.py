import ast
import sys
from typing import Any
from .internal import Filters, Key
def _handle_logical_op(node) -> Filters:
    op = 'AND' if isinstance(node.op, ast.And) else 'OR'
    filters = [_parse_node(n) for n in node.values]
    return Filters(op=op, filters=filters)