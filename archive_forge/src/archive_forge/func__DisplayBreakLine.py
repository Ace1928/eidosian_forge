from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from functools import partial
from apitools.base.py import encoding
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import text
from sqlparse import lexer
from sqlparse import tokens as T
def _DisplayBreakLine(self, out, prepend, beneath_stub, is_root):
    """Displays an empty line between nodes for visual breathing room.

    Keeps in tact the vertical lines connecting all immediate children of a
    node to each other.

    Args:
      out: Output stream to which we print.
      prepend: String that precedes any information about this node to maintain
        a visible hierarchy.
      beneath_stub: String that preserves the indentation of the vertical lines.
      is_root: Boolean indicating whether this node is the root of the tree.
    """
    above_child = '  ' if is_root else ''
    above_child += '  |' if self.children else ''
    break_line = '{}{}{}'.format(prepend, beneath_stub, above_child)
    out.Print(break_line.rstrip())