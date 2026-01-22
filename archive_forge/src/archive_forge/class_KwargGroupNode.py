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
class KwargGroupNode(LayoutNode):
    """
  A keyword argument group. Contains a keyword, followed by an argument group.
  """

    def __init__(self, pnode):
        super(KwargGroupNode, self).__init__(pnode)
        self._layout_passes = [(0, False), (1, False), (2, False), (3, False), (4, False), (5, True)]

    def has_terminal_comment(self):
        if self.children[-1].node_type is NodeType.COMMENT:
            return True
        return self.children[-1].has_terminal_comment()

    @property
    def name(self):
        return self.children[0].pnode.children[0].spelling.upper()

    def _validate_layout(self, stack_context, start_extent, end_extent):
        config = stack_context.config
        if end_extent[1] > config.format.linewidth:
            return False
        size = end_extent - start_extent
        if not self._wrap:
            if size[0] > 1 and len(self.name) > config.format.max_prefix_chars:
                return False
        return True

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        start_cursor = cursor.clone()
        self._colextent = cursor[1]
        assert len(self.children) <= 2, 'Expected at most 2 children in KWargGroup, got {}'.format(len(self.children))
        child = self.children[0]
        assert child.node_type == NodeType.KEYWORD
        if len(self.name) <= config.format.min_prefix_chars:
            self._wrap = False
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = child.colextent
        if len(self.children) == 1:
            return cursor
        if self._wrap:
            column_cursor = Cursor(cursor[0] + 1, start_cursor[1] + config.format.tab_size)
        else:
            column_cursor = cursor + (0, 1)
        child = self.children[1]
        cursor = Cursor(*column_cursor)
        if self.statement_terminal:
            child.statement_terminal = True
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        column_cursor[0] = cursor[0]
        return cursor