from __future__ import print_function
from __future__ import unicode_literals
import logging
from cmakelang import lex
from cmakelang.common import InternalError
from cmakelang.parse.printer import dump_tree_tostr
from cmakelang.parse.util import (
from cmakelang.parse.common import (
from cmakelang.parse.simple_nodes import CommentNode, OnOffNode
class StandardParser2(object):

    def __init__(self, cmdspec=None, funtree=None, doc=None):
        if cmdspec is None:
            cmdspec = ('<none>', '*')
        if funtree is None:
            funtree = {}
        self.cmdspec = cmdspec
        self.funtree = funtree
        self.doc = doc

    @property
    def pspec(self):
        return self.cmdspec.pargs

    @property
    def kwargs(self):
        return self.funtree

    def __call__(self, ctx, tokens, breakstack):
        return StandardArgTree.parse2(ctx, tokens, self.cmdspec, self.funtree, breakstack)