from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
from cmakelang.parse.argument_nodes import (
class FlagGroupNode(PositionalGroupNode):
    """A positinal group where each argument is a flag."""

    @classmethod
    def parse(cls, ctx, tokens, flags, breakstack):
        """
    Parse a continuous sequence of flags
    """
        tree = cls()
        tree.spec = PositionalSpec('+', False, [], flags)
        while tokens:
            if should_break(tokens[0], breakstack):
                break
            if tokens[0].type in WHITESPACE_TOKENS:
                tree.children.append(tokens.pop(0))
                continue
            if tokens[0].spelling.upper() not in flags:
                break
            child = TreeNode(NodeType.FLAG)
            child.children.append(tokens.pop(0))
            CommentNode.consume_trailing(ctx, tokens, child)
            tree.children.append(child)
        return tree