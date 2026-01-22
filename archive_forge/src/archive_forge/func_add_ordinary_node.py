import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def add_ordinary_node(self, ast_node):
    """Grows the graph by adding an ordinary CFG node.

    Ordinary nodes are followed by the next node, in lexical order, that is,
    they become the new leaf set.

    Args:
      ast_node: ast.AST

    Returns:
      Node
    """
    node = self._add_new_node(ast_node)
    self.leaves = set((node,))
    return node