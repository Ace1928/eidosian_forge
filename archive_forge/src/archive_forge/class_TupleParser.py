from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
from cmakelang.parse.argument_nodes import (
class TupleParser(object):

    def __init__(self, ntup, npargs=None, flags=None):
        if npargs is None:
            npargs = '*'
        if flags is None:
            flags = []
        self.npargs = npargs
        self.ntup = ntup
        self.flags = flags

    def __call__(self, ctx, tokens, breakstack):
        return TupleGroupNode.parse(ctx, tokens, self.npargs, self.ntup, self.flags, breakstack)