from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
from cmakelang.parse.argument_nodes import (
class ShellCommandNode(StandardArgTree):
    """Shell commands are children of a `COMMAND` keyword argument and are
     common enough to warrant their own node. We also will likely want some
     special formatting rules for these nodes.
  """

    @classmethod
    def parse(cls, ctx, tokens, breakstack):
        """
    Parser for the COMMAND kwarg lists in the form of::

        COMMAND foo --long-flag1 arg1 arg2 --long-flag2 -a -b -c arg3 arg4

    The parser acts very similar to a standard parser where `--xxx` is treated
    as a keyword argument and `-x` is treated as a flag.
    """
        tree = super(ShellCommandNode, cls).parse(ctx, tokens, '+', {}, [], breakstack)
        for pgroup in tree.parg_groups:
            pgroup.tags.append('cmdline')
        return tree