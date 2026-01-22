import sys
import ast
import py
from py._code.assertion import _format_explanation, BuiltinAssertionError
def _is_ast_expr(node):
    return isinstance(node, ast.expr)