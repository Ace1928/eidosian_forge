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
class AtWordStatementNode(LayoutNode):
    """
  Top-level node for an atword representing a statement. Such as
  @PACKAGE_INIT@
  """

    def __init__(self, pnode):
        super(AtWordStatementNode, self).__init__(pnode)
        self._layout_passes = [(0, False)]

    def reflow(self, stack_context, cursor, _=0):
        return super(AtWordStatementNode, self).reflow(stack_context, cursor, max((passno for passno, _ in self._layout_passes)))

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        start_cursor = cursor.clone()
        self._colextent = cursor[1]
        children = list(self.children)
        assert children
        child = children.pop(0)
        assert child.node_type == NodeType.ATWORD
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        if children:
            cursor[1] += 1
            child = children.pop(0)
            assert child.node_type == NodeType.COMMENT, 'Expected COMMENT after RPAREN but got {}'.format(child.node_type)
            assert not children
            savecursor = cursor.clone()
            cursor = child.reflow(stack_context, cursor, passno)
            if child.colextent > config.format.linewidth:
                cursor = child.reflow(stack_context, Cursor(savecursor[0] + 1, start_cursor[1]), passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
        return cursor

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        super(AtWordStatementNode, self).write(config, ctx)