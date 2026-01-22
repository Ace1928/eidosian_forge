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
class PargGroupNode(LayoutNode):
    """
  A group of positional arguments.
  """

    def __init__(self, pnode):
        super(PargGroupNode, self).__init__(pnode)
        self._max_pargs_hwrap = None
        self._layout_passes = [(0, False), (1, False), (2, False), (3, False), (4, True)]

    def has_terminal_comment(self):
        if not self.children:
            return False
        if self.children[-1].node_type is NodeType.COMMENT:
            return True
        return self.children[-1].has_terminal_comment()

    def lock(self, config, stmt_depth=0):
        if config.format.autosort and isinstance(self.pnode, PositionalGroupNode) and self.pnode.sortable:
            self._children = sort_arguments(self._children)
        super(PargGroupNode, self).lock(config, stmt_depth)
        self._max_pargs_hwrap = config.format.max_pargs_hwrap
        if isinstance(self.pnode, PositionalGroupNode) and self.pnode.spec is not None and (self.pnode.spec.max_pargs_hwrap is not None):
            self._max_pargs_hwrap = self.pnode.spec.max_pargs_hwrap

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        children = list(self.children)
        self._colextent = cursor.y
        prev = None
        child = None
        column_cursor = cursor.clone()
        numpargs = count_arguments(children)
        is_cmdline = 'cmdline' in self.pnode.tags
        if is_cmdline:
            self._wrap = False
        rowcount = 0
        while children:
            prev = child
            child = children.pop(0)
            cursor_is_at_column = True
            if prev is None:
                rowcount = 1
            elif is_line_comment(prev) or prev.has_terminal_comment() or self._wrap:
                column_cursor[0] = cursor[0] + 1
                cursor = Cursor(*column_cursor)
                rowcount += 1
            else:
                cursor_is_at_column = False
                cursor[1] += 1
            if self.statement_terminal and (not children):
                child.statement_terminal = True
            input_cursor = cursor
            cursor = child.reflow(stack_context, input_cursor, passno)
            if not cursor_is_at_column and (not self._wrap):
                needs_wrap = False
                if child.colextent > config.format.linewidth:
                    needs_wrap = True
                if self.statement_terminal:
                    if cursor[1] + 1 > config.format.linewidth:
                        needs_wrap = True
                if (prev and prev.node_type in SCALAR_TYPES) and child.node_type is NodeType.COMMENT and (not child.is_tag()):
                    needs_wrap = True
                if needs_wrap:
                    rowcount += 1
                    column_cursor[0] = input_cursor[0] + 1
                    cursor = child.reflow(stack_context, column_cursor, passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
        self._rowextent = rowcount
        if is_cmdline:
            self._reflow_valid &= rowcount <= config.format.max_rows_cmdline
        elif numpargs > self._max_pargs_hwrap:
            self._reflow_valid &= self._wrap
        return cursor