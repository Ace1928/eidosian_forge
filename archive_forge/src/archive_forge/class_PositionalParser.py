from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class PositionalParser(object):

    def __init__(self, npargs=None, flags=None, sortable=False):
        if npargs is None:
            npargs = '*'
        if flags is None:
            flags = []
        self.npargs = npargs
        self.flags = flags
        self.sortable = sortable

    def __call__(self, ctx, tokens, breakstack):
        return PositionalGroupNode.parse(ctx, tokens, self.npargs, self.flags, breakstack, self.sortable)