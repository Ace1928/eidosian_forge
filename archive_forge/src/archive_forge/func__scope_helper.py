from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import contextlib
import itertools
import tokenize
from six import StringIO
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
def _scope_helper(node):
    """Get the closure of nodes that could begin a scope at this point.

  For instance, when encountering a `(` when parsing a BinOp node, this could
  indicate that the BinOp itself is parenthesized OR that the BinOp's left node
  could be parenthesized.

  E.g.: (a + b * c)   or   (a + b) * c   or   (a) + b * c
        ^                  ^                  ^

  Arguments:
    node: (ast.AST) Node encountered when opening a scope.

  Returns:
    A closure of nodes which that scope might apply to.
  """
    if isinstance(node, ast.Attribute):
        return (node,) + _scope_helper(node.value)
    if isinstance(node, ast.Subscript):
        return (node,) + _scope_helper(node.value)
    if isinstance(node, ast.Assign):
        return (node,) + _scope_helper(node.targets[0])
    if isinstance(node, ast.AugAssign):
        return (node,) + _scope_helper(node.target)
    if isinstance(node, ast.Expr):
        return (node,) + _scope_helper(node.value)
    if isinstance(node, ast.Compare):
        return (node,) + _scope_helper(node.left)
    if isinstance(node, ast.BoolOp):
        return (node,) + _scope_helper(node.values[0])
    if isinstance(node, ast.BinOp):
        return (node,) + _scope_helper(node.left)
    if isinstance(node, ast.Tuple) and node.elts:
        return (node,) + _scope_helper(node.elts[0])
    if isinstance(node, ast.Call):
        return (node,) + _scope_helper(node.func)
    if isinstance(node, ast.GeneratorExp):
        return (node,) + _scope_helper(node.elt)
    if isinstance(node, ast.IfExp):
        return (node,) + _scope_helper(node.body)
    return (node,)