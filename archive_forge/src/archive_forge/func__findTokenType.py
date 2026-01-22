from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from antlr3.constants import INVALID_TOKEN_TYPE
from antlr3.tokens import CommonToken
from antlr3.tree import CommonTree, CommonTreeAdaptor
import six
from six.moves import range
def _findTokenType(self, t, ttype):
    """Return a List of tree nodes with token type ttype"""
    nodes = []

    def visitor(tree, parent, childIndex, labels):
        nodes.append(tree)
    self.visit(t, ttype, visitor)
    return nodes