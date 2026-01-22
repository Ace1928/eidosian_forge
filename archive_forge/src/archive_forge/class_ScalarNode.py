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
class ScalarNode(LayoutNode):
    """
  Holds scalar tokens such as statement names, parentheses, or keyword or
  positional arguments.
  """

    def has_terminal_comment(self):
        if not self.children:
            return False
        if not self.children[-1].pnode.children:
            return False
        return self.children[-1].pnode.children[0].type == TokenType.COMMENT

    def _reflow(self, stack_context, cursor, passno):
        """
    Reflow is pretty trivial for a scalar node. We don't have any choices to
    make, there is only one possible rendering.
    """
        assert self.pnode.children
        token = self.pnode.children[0]
        assert isinstance(token, lex.Token)
        lines = normalize_line_endings(token.spelling).split('\n')
        line = lines.pop(0)
        cursor[1] += len(line)
        self._colextent = cursor[1]
        while lines:
            cursor[0] += 1
            cursor[1] = 0
            line = lines.pop(0)
            cursor[1] += len(line)
            self._colextent = max(self._colextent, cursor[1])
        if self.children:
            assert len(self.children) == 1
            child = self.children[0]
            assert child.node_type == NodeType.COMMENT
            cursor = child.reflow(stack_context, cursor + (0, 1), passno)
            self._reflow_valid &= child.reflow_valid
            self._colextent = max(self._colextent, child.colextent)
        return cursor

    def write(self, config, ctx):
        if not ctx.is_active():
            return
        assert self.pnode.children
        token = self.pnode.children[0]
        assert isinstance(token, lex.Token)
        spelling = normalize_line_endings(token.spelling)
        if self.node_type == NodeType.FUNNAME:
            command_case = config.resolve_for_command(token.spelling, 'format.command_case')
            if command_case in ('lower', 'upper'):
                spelling = getattr(token.spelling, command_case)()
            elif command_case == 'canonical':
                if self._parent.pnode.cmdspec is not None and self._parent.pnode.cmdspec.spelling is not None:
                    spelling = self._parent.pnode.cmdspec.spelling
                else:
                    spelling = config.resolve_for_command(token.spelling, 'spelling', token.spelling.lower())
            else:
                assert command_case == 'unchanged', 'Unrecognized command case {}'.format(command_case)
        elif self.node_type in (NodeType.KEYWORD, NodeType.FLAG) and config.format.keyword_case in ('lower', 'upper'):
            spelling = getattr(token.spelling, config.format.keyword_case)()
        ctx.outfile.write_at(self.position, spelling)
        children = list(self.children)
        if children:
            child = children.pop(0)
            assert child.node_type == NodeType.COMMENT
            child.write(config, ctx)
        assert not children