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
class StatementNode(LayoutNode):
    """
  Top-level node for a statement.
  """

    def __init__(self, pnode):
        super(StatementNode, self).__init__(pnode)
        self._layout_passes = [(0, False), (1, True), (2, True), (3, True), (4, True), (5, True)]

    def reflow(self, stack_context, cursor, _=0):
        return super(StatementNode, self).reflow(stack_context, cursor, max((passno for passno, _ in self._layout_passes)))

    def get_prefix_width(self, config):
        prefix_width = len(self.name) + 1
        if need_paren_space(self.name, config):
            prefix_width += 1
        return prefix_width

    @property
    def name(self):
        return self.children[0].pnode.children[0].spelling.lower()

    def _validate_layout(self, stack_context, start_extent, end_extent):
        config = stack_context.config
        if end_extent[1] > config.format.linewidth:
            return False
        size = end_extent - start_extent
        if not self._wrap:
            if size[0] > 1 and self.get_prefix_width(config) > config.format.max_prefix_chars:
                return False
        return True

    def _reflow(self, stack_context, cursor, passno):
        config = stack_context.config
        start_cursor = cursor.clone()
        self._colextent = cursor[1]
        children = list(self.children)
        assert children
        child = children.pop(0)
        assert child.node_type == NodeType.FUNNAME
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        funname = child
        token = funname.pnode.children[0]
        assert isinstance(token, lex.Token)
        if need_paren_space(token.spelling.lower(), config):
            cursor[1] += 1
        if self.get_prefix_width(config) <= config.format.min_prefix_chars:
            self._wrap = False
        assert children
        child = children.pop(0)
        assert child.node_type == NodeType.LPAREN
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        if self._wrap:
            column_cursor = start_cursor + (1, config.format.tab_size)
            cursor = Cursor(*column_cursor)
        else:
            column_cursor = cursor.clone()
        child = children.pop(0)
        assert child.node_type is NodeType.ARGGROUP, 'Expected ARGGROUP node, but got {} at {}'.format(child.node_type, child.pnode)
        child.statement_terminal = True
        cursor = child.reflow(stack_context, cursor, passno)
        self._reflow_valid &= child.reflow_valid
        self._colextent = max(self._colextent, child.colextent)
        column_cursor[0] = cursor[0]
        assert children
        prev = child
        child = children.pop(0)
        assert child.node_type == NodeType.RPAREN, 'Expected RPAREN but got {}'.format(child.node_type)
        dangle_parens = False
        if config.format.dangle_parens and cursor[0] > start_cursor[0]:
            dangle_parens = True
        elif cursor[1] >= config.format.linewidth:
            dangle_parens = True
            if not self._wrap:
                self._reflow_valid = False
        elif prev.has_terminal_comment():
            dangle_parens = True
        column_cursor[0] += 1
        if config.format.dangle_align == 'prefix':
            dangle_cursor = Cursor(column_cursor[0], start_cursor[1])
        elif config.format.dangle_align == 'prefix-indent':
            dangle_cursor = Cursor(column_cursor[0], start_cursor[1] + config.format.tab_size)
        elif config.format.dangle_align == 'child':
            dangle_cursor = Cursor(*column_cursor)
        else:
            raise ValueError('Unexpected config.format.dangle_align: {}'.format(config.format.dangle_align))
        if dangle_parens:
            cursor = dangle_cursor.clone()
        rparen = child
        initial_rparen_cursor = cursor.clone()
        cursor = rparen.reflow(stack_context, initial_rparen_cursor, passno)
        self._reflow_valid &= rparen.reflow_valid
        if children:
            cursor[1] += 1
            child = children.pop(0)
            assert child.node_type == NodeType.COMMENT, 'Expected COMMENT after RPAREN but got {}'.format(child.node_type)
            assert not children
            savecursor = cursor.clone()
            cursor = child.reflow(stack_context, cursor, passno)
            if not dangle_parens and child.colextent > config.format.linewidth:
                cursor = rparen.reflow(stack_context, dangle_cursor, passno)
                self._reflow_valid &= rparen.reflow_valid
                savecursor = cursor.clone()
                cursor = child.reflow(stack_context, cursor, passno)
            if child.colextent > config.format.linewidth:
                cursor = rparen.reflow(stack_context, initial_rparen_cursor, passno)
                cursor = child.reflow(stack_context, Cursor(savecursor[0] + 1, start_cursor[1]), passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
        self._colextent = max(self._colextent, rparen.colextent)
        return cursor

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        super(StatementNode, self).write(config, ctx)