import ast
import sys
from typing import Any
from .internal import Filters, Key
def _handle_comparison(node) -> Filters:
    op_map = {'Gt': '>', 'Lt': '<', 'Eq': '==', 'NotEq': '!=', 'GtE': '>=', 'LtE': '<=', 'In': 'IN', 'NotIn': 'NIN'}
    left_operand = node.left.id if isinstance(node.left, ast.Name) else None
    left_operand_mapped = to_frontend_name(left_operand)
    right_operand = _extract_value(node.comparators[0])
    operation = type(node.ops[0]).__name__
    return Filters(op=op_map.get(operation), key=_server_path_to_key(left_operand) if left_operand_mapped else None, value=right_operand, disabled=False)