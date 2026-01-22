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
class FlowControlNode(LayoutNode):
    """
  Top-Level node composed of a flow-control statement and it's associated
  `BodyNodes`.
  """

    def _reflow(self, stack_context, cursor, passno):
        """
    Compute the size of a flowcontrol block
    """
        config = stack_context.config
        self._colextent = 0
        column_cursor = cursor.clone()
        children = list(self.children)
        assert children
        child = children.pop(0)
        assert child.node_type == NodeType.STATEMENT
        cursor = child.reflow(stack_context, column_cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        column_cursor[0] = cursor[0] + 1
        assert children
        child = children.pop(0)
        assert child.node_type == NodeType.BODY
        cursor = child.reflow(stack_context, column_cursor + (0, config.format.tab_size), passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        column_cursor[0] = cursor[0] + 1
        while True:
            assert children
            child = children.pop(0)
            assert child.node_type == NodeType.STATEMENT
            cursor = child.reflow(stack_context, column_cursor, passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
            column_cursor[0] = cursor[0] + 1
            if not children:
                break
            child = children.pop(0)
            assert child.node_type == NodeType.BODY
            cursor = child.reflow(stack_context, column_cursor + (0, config.format.tab_size), passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
            column_cursor[0] = cursor[0] + 1
        return cursor