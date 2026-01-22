from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import io
import logging
import re
import sys
from cmakelang import lex
from cmakelang import markup
from cmakelang.common import UserError
from cmakelang.lex import TokenType
from cmakelang.parse.argument_nodes import PositionalGroupNode
from cmakelang.parse.common import FlowType, NodeType, TreeNode
from cmakelang.parse.util import comment_is_tag
from cmakelang.parse import simple_nodes
class ArgGroupNode(LayoutNode):
    """
  A group of arguments. This is the single child node of either a
  `StatementNode` or `KwargGroupNode` which then contains any further
  group nodes.
  """

    def __init__(self, pnode):
        super(ArgGroupNode, self).__init__(pnode)
        self._max_subgroups_hwrap = None
        self._layout_passes = [(0, False), (1, False), (2, False), (3, False), (4, True), (5, True)]

    def lock(self, config, stmt_depth=0):
        super(ArgGroupNode, self).lock(config, stmt_depth)
        self._max_subgroups_hwrap = config.format.max_subgroups_hwrap
        if hasattr(self.pnode, 'cmdspec') and getattr(self.pnode, 'cmdspec') is not None and (getattr(self.pnode, 'cmdspec').max_subgroups_hwrap is not None):
            self._max_subgroups_hwrap = getattr(self.pnode, 'cmdspec').max_subgroups_hwrap

    def has_terminal_comment(self):
        """
    An ArgGroup is a container for one or more PARGGROUP, FLAGGROUP, or
    KWARGGROUP subtrees. Any terminal comment will belong to one of
    it's children.
    """
        return self.children and (self.children[-1].node_type is NodeType.COMMENT or self.children[-1].has_terminal_comment())

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        children = list(self.children)
        self._colextent = cursor.y
        prev = None
        child = None
        column_cursor = cursor.clone()
        numgroups = count_subgroups(children)
        while children:
            prev = child
            child = children.pop(0)
            if prev is None:
                is_first_in_row = True
            elif is_line_comment(prev) or prev.has_terminal_comment() or self._wrap:
                column_cursor[0] += 1
                cursor = column_cursor.clone()
                is_first_in_row = True
            else:
                cursor[1] += 1
                is_first_in_row = False
            if self.statement_terminal and (not children):
                child.statement_terminal = True
            start_cursor = cursor
            cursor = child.reflow(stack_context, cursor, passno)
            if not is_first_in_row and (not self._wrap):
                needs_wrap = False
                if child.statement_terminal:
                    if cursor[1] + 1 > config.format.linewidth:
                        needs_wrap = True
                if child.colextent > config.format.linewidth:
                    needs_wrap = True
                if (cursor - start_cursor)[0] > 1:
                    needs_wrap = True
                if needs_wrap:
                    column_cursor[0] += 1
                    cursor = Cursor(*column_cursor)
                    cursor = child.reflow(stack_context, cursor, passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
            column_cursor[0] = cursor[0]
        if numgroups > self._max_subgroups_hwrap:
            self._reflow_valid &= self._wrap
        return cursor

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        super(ArgGroupNode, self).write(config, ctx)